"""BiomedCCPL model: VGAP, SCD, and semantic anchor regularization."""

import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import CUSTOM_TEMPLATES
from open_clip import create_model_from_pretrained, get_tokenizer


class LayerNorm(nn.LayerNorm):
    """Compute LayerNorm in fp32 for stable mixed-precision training."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CustomCLIP(nn.Module):
    """BiomedCLIP with image-conditional causal/non-causal prompt pathways."""

    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_ctx = 4
        self.biomedclip_model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype
        self.logit_scale = biomedclip_model.logit_scale

        self.alpha = cfg.TRAINER.BIOMEDCCPL.get("ALPHA", 0.01)
        self.protonum = cfg.TRAINER.BIOMEDCCPL.get("PROTONUM", 14)
        self.lambda_nem = cfg.TRAINER.BIOMEDCCPL.get("LAMBDA_NEM", 0.1)
        self.lambda_ortho = cfg.TRAINER.BIOMEDCCPL.get("LAMBDA_ORTHO", 0.01)
        self.crosslayers = cfg.TRAINER.BIOMEDCCPL.get("CROSSLAYERS", [3, 7, 11])

        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        with torch.no_grad():
            device = self.biomedclip_model.logit_scale.device
            tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            anchor_model = self.biomedclip_model.float().eval().to(device)

            clean_classnames = [name.replace("_", " ") for name in classnames]

            hand_template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            tokenized_prompts = torch.cat(
                [tokenizer(hand_template.format(c)) for c in clean_classnames]
            ).to(device)
            hand_feas = anchor_model.encode_text(tokenized_prompts)
            # Anchors are regenerated from the active dataset class names.
            self.register_buffer(
                "hand_feas",
                hand_feas / hand_feas.norm(dim=-1, keepdim=True),
                persistent=False,
            )

            l_tokenized_prompts = torch.cat(
                [tokenizer(hand_template.format(c)) for c in clean_classnames]
            ).to(device)
            self.register_buffer(
                "l_tokenized_prompts", l_tokenized_prompts, persistent=False
            )
            x = anchor_model.text.transformer.embeddings.word_embeddings(
                self.l_tokenized_prompts
            )
            self.register_buffer("prefix", x[:, :1, :], persistent=False)
            ctx_vectors = x[0, 1 : 1 + self.n_ctx, :].clone().detach()
            self.register_buffer("suffix", x[:, 1 + self.n_ctx :, :], persistent=False)

        self.ctx_causal = nn.Parameter(ctx_vectors.clone())
        noise = torch.randn_like(ctx_vectors) * 0.01
        self.ctx_non_causal = nn.Parameter(ctx_vectors.clone() + noise)

        # B, 196, 768 -> B, 196, 14
        single_prototype_learner = nn.Sequential(
            nn.Linear(768, 256, bias=False),
            LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.protonum, bias=False),
        )
        self.prototype_learner = nn.ModuleList(
            [
                copy.deepcopy(single_prototype_learner)
                for _ in range(len(self.crosslayers))
            ]
        )
        single_crossattention_layer = nn.MultiheadAttention(
            embed_dim=768, num_heads=768 // 64, batch_first=True
        )
        self.crossattention_layer = nn.ModuleList(
            [
                copy.deepcopy(single_crossattention_layer)
                for _ in range(len(self.crosslayers))
            ]
        )
        self.crossattention_layer_nc = nn.ModuleList(
            [
                copy.deepcopy(single_crossattention_layer)
                for _ in range(len(self.crosslayers))
            ]
        )

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        batch_size = image.shape[0]

        content_causal = self.ctx_causal.unsqueeze(0).expand(self.n_cls, -1, -1)
        word_emb_causal = torch.cat([self.prefix, content_causal, self.suffix], dim=1)
        text_emb_causal = self._get_text_embedding_base(word_emb_causal)
        seq_len = text_emb_causal.shape[1]
        text_emb_causal = (
            text_emb_causal.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self.n_cls, seq_len, -1)
        )

        if self.training:
            content_non_causal = self.ctx_non_causal.unsqueeze(0).expand(
                self.n_cls, -1, -1
            )
            word_emb_non_causal = torch.cat(
                [self.prefix, content_non_causal, self.suffix], dim=1
            )
            text_emb_non_causal = self._get_text_embedding_base(word_emb_non_causal)
            text_emb_non_causal = (
                text_emb_non_causal.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .reshape(batch_size * self.n_cls, seq_len, -1)
            )

        attention_mask_after = (self.l_tokenized_prompts != 0).long().to(image.device)
        attention_mask_after = attention_mask_after.repeat(batch_size, 1)
        extended_attention_mask = self._get_attention_mask(attention_mask_after)

        image_emb = self.biomedclip_model.visual.trunk.patch_embed(image)
        image_emb = torch.cat(
            (
                self.biomedclip_model.visual.trunk.cls_token.expand(
                    image_emb.shape[0], -1, -1
                ),
                image_emb,
            ),
            dim=1,
        )
        image_emb = image_emb + self.biomedclip_model.visual.trunk.pos_embed

        counter = 0
        for i in range(12):
            text_emb_causal = self.biomedclip_model.text.transformer.encoder.layer[i](
                text_emb_causal, attention_mask=extended_attention_mask
            )[0]
            if self.training:
                text_emb_non_causal = (
                    self.biomedclip_model.text.transformer.encoder.layer[i](
                        text_emb_non_causal, attention_mask=extended_attention_mask
                    )[0]
                )
            image_emb = self.biomedclip_model.visual.trunk.blocks[i](image_emb)
            if i in self.crosslayers:
                w_score = self.prototype_learner[counter](image_emb[:, 1:, :])
                w_score_softmax = F.softmax(w_score, dim=1)
                # B, 14, 196 @ B, 196, 768 -> B, 14, 768
                prototype = torch.matmul(
                    w_score_softmax.transpose(1, 2), image_emb[:, 1:, :]
                )

                text_emb_causal_batch = text_emb_causal.reshape(
                    batch_size, self.n_cls, seq_len, -1
                )

                crossatt_out_c, _ = self.crossattention_layer[counter](
                    query=text_emb_causal_batch[:, :, 0, :],  # (B, ncls, 768)
                    key=prototype,  # (B, protonum, 768)
                    value=prototype,
                )
                cau_class_feature = (
                    self.alpha * text_emb_causal_batch[:, :, 0, :]
                    + (1 - self.alpha) * crossatt_out_c
                )
                text_emb_causal = torch.cat(
                    [
                        cau_class_feature.unsqueeze(2),
                        text_emb_causal_batch[:, :, 1:, :],
                    ],
                    dim=2,
                ).reshape(batch_size * self.n_cls, seq_len, -1)

                if self.training:
                    text_emb_non_causal_batch = text_emb_non_causal.reshape(
                        batch_size, self.n_cls, seq_len, -1
                    )
                    crossatt_out_u, _ = self.crossattention_layer_nc[counter](
                        query=text_emb_non_causal_batch[:, :, 0, :],
                        key=prototype,
                        value=prototype,
                    )
                    adv_class_feature = (
                        self.alpha * text_emb_non_causal_batch[:, :, 0, :]
                        + (1 - self.alpha) * crossatt_out_u
                    )
                    text_emb_non_causal = torch.cat(
                        [
                            adv_class_feature.unsqueeze(2),
                            text_emb_non_causal_batch[:, :, 1:, :],
                        ],
                        dim=2,
                    ).reshape(batch_size * self.n_cls, seq_len, -1)

                counter += 1

        cau_text_feature = self._get_final_text_features(text_emb_causal).reshape(
            batch_size, self.n_cls, -1
        )
        image_features = self._get_final_image_features(image_emb)
        logits = logit_scale * torch.einsum(
            "bd,bcd->bc", image_features, cau_text_feature
        )

        if self.training:
            loss_ce = F.cross_entropy(logits, label)
            loss_sar = self._compute_sar_loss(cau_text_feature)

            adv_text_features = self._get_final_text_features(
                text_emb_non_causal
            ).reshape(batch_size, self.n_cls, -1)

            ortho_matrix = torch.matmul(
                F.normalize(cau_text_feature, dim=-1),
                F.normalize(adv_text_features, dim=-1).transpose(1, 2),
            )
            loss_ortho = torch.mean(ortho_matrix**2)

            logits_adv = logit_scale * torch.einsum(
                "bd,bcd->bc", image_features, adv_text_features
            )
            log_probs_adv = F.log_softmax(logits_adv, dim=-1)
            uniform_dist = torch.full_like(log_probs_adv, 1.0 / self.n_cls)
            loss_nem = F.kl_div(
                log_probs_adv, uniform_dist, reduction="batchmean"
            )  # Make the non-causal pathway non-predictive.

            total_loss = (
                loss_ce
                + self.lambda_ortho * loss_ortho
                + self.lambda_nem * loss_nem
                + loss_sar
            )
            return logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar
        else:
            return logits

    def _get_text_embedding_base(self, word_emb):
        ncls, seq_len, _ = word_emb.shape
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=word_emb.device)
            .unsqueeze(0)
            .expand(ncls, -1)
        )
        position_emb = (
            self.biomedclip_model.text.transformer.embeddings.position_embeddings(
                position_ids
            )
        )
        token_type_ids = torch.zeros(
            ncls, seq_len, dtype=torch.long, device=word_emb.device
        )
        token_type_emb = (
            self.biomedclip_model.text.transformer.embeddings.token_type_embeddings(
                token_type_ids
            )
        )
        embeddings = word_emb + position_emb + token_type_emb
        text_emb = self.biomedclip_model.text.transformer.embeddings.LayerNorm(
            embeddings
        )
        return self.biomedclip_model.text.transformer.embeddings.dropout(text_emb)

    def _get_attention_mask(self, attention_mask):
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=self.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def _get_final_text_features(self, text_emb):
        cls_features = text_emb[:, 0]
        text_features_proj = self.biomedclip_model.text.proj(cls_features)
        return text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)

    def _get_final_image_features(self, image_emb):
        image_emb_norm = self.biomedclip_model.visual.trunk.norm(image_emb)
        image_features_proj = self.biomedclip_model.visual.head.proj(
            image_emb_norm[:, 0]
        )
        return image_features_proj / image_features_proj.norm(dim=-1, keepdim=True)

    def _compute_sar_loss(self, text_features):
        target = self.hand_feas
        if text_features.dim() == 3:
            target = target.unsqueeze(0).expand_as(text_features)
        return F.mse_loss(text_features, target)


@TRAINER_REGISTRY.register()
class BIOMEDCCPL_BiomedCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCCPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        if (
            cfg.TRAINER.BIOMEDCCPL.PREC == "fp32"
            or cfg.TRAINER.BIOMEDCCPL.PREC == "amp"
        ):
            biomedclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = [
            "ctx_causal",
            "ctx_non_causal",
            "prototype_learner",
            "crossattention_layer",
        ]
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if any(re_name in name for re_name in names_to_update):
                param.requires_grad_(True)
        enabled = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("causal_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCCPL.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.BIOMEDCCPL.PREC

        if prec == "amp":
            with autocast():
                logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar = model(
                    image, label
                )
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, total_loss, loss_ce, loss_ortho, loss_nem, loss_sar = model(
                image, label
            )
            self.model_backward_and_update(total_loss)

        loss_summary = {
            "total_loss": total_loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_ortho": loss_ortho.item(),
            "loss_nem": loss_nem.item(),
            "loss_sar": loss_sar.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        # Step the epoch-level learning-rate scheduler after the final batch.
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print(f"Loading weights to {name} from '{model_path}' (epoch = {epoch})")

            self._models[name].load_state_dict(state_dict, strict=False)
