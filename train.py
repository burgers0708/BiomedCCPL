"""Training and evaluation entry point for BiomedCCPL."""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from yacs.config import CfgNode as CN

import datasets  # noqa: F401 - registers the supported datasets
import trainers  # noqa: F401 - registers the BiomedCCPL trainer
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger


def extend_cfg(cfg):
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.BIOMEDCCPL = CN()
    cfg.TRAINER.BIOMEDCCPL.PREC = "fp32"
    cfg.TRAINER.BIOMEDCCPL.ALPHA = 1.0
    cfg.TRAINER.BIOMEDCCPL.PROTONUM = 14
    cfg.TRAINER.BIOMEDCCPL.LAMBDA_NEM = 0.1
    cfg.TRAINER.BIOMEDCCPL.LAMBDA_ORTHO = 0.01
    cfg.TRAINER.BIOMEDCCPL.CROSSLAYERS = [3, 7, 11]


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def print_configuration(args, cfg):
    print("Arguments")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("Configuration")
    print(cfg)


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    print_configuration(args, cfg)
    print(f"System information\n{collect_env_info()}\n")
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    elif not args.no_train:
        trainer.train()


def build_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate BiomedCCPL")
    parser.add_argument("--root", default="", help="dataset root directory")
    parser.add_argument("--output-dir", default="", help="experiment output directory")
    parser.add_argument("--resume", default="", help="directory containing checkpoints")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config-file", default="", help="BiomedCCPL YAML config")
    parser.add_argument("--trainer", default="BIOMEDCCPL_BiomedCLIP")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--model-dir", default="", help="checkpoint root for evaluation"
    )
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="configuration overrides, e.g. DATASET.NAME BTMRI",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
