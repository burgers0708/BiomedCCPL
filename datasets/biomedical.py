"""Unified loader for the 11 biomedical classification datasets."""

import math
import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing, read_json, write_json


SUPPORTED_DATASETS = (
    "BTMRI",
    "BUSI",
    "CHMNIST",
    "COVID_19",
    "CTKidney",
    "DermaMNIST",
    "KneeXray",
    "Kvasir",
    "LungColon",
    "OCTMNIST",
    "RETINA",
)


class BiomedicalDataset(DatasetBase):
    """Folder-based dataset with deterministic few-shot split caching."""

    dataset_dir = None

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_root = os.path.join(root, self.dataset_dir)
        image_dir = os.path.join(dataset_root, self.dataset_dir)
        split_path = os.path.join(dataset_root, f"split_{self.dataset_dir}.json")
        fewshot_dir = os.path.join(dataset_root, "split_fewshot")
        mkdir_if_missing(fewshot_dir)

        if os.path.exists(split_path):
            train, val, test = self.read_split(split_path, image_dir)
        else:
            train, val, test = self.read_and_split_data(image_dir)
            self.save_split(train, val, test, split_path, image_dir)

        if cfg.DATASET.NUM_SHOTS >= 1:
            cache_path = os.path.join(
                fewshot_dir,
                f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pkl",
            )
            if os.path.exists(cache_path):
                print(f"Loading few-shot split from {cache_path}")
                with open(cache_path, "rb") as stream:
                    cached = pickle.load(stream)
                train, val = cached["train"], cached["val"]
            else:
                train = self.generate_fewshot_dataset(
                    train, num_shots=cfg.DATASET.NUM_SHOTS
                )
                val = self.generate_fewshot_dataset(
                    val, num_shots=min(cfg.DATASET.NUM_SHOTS, 4)
                )
                with open(cache_path, "wb") as stream:
                    pickle.dump(
                        {"train": train, "val": val},
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"Saved few-shot split to {cache_path}")

        train, val, test = self.subsample_classes(
            train,
            val,
            test,
            subsample=cfg.DATASET.SUBSAMPLE_CLASSES,
        )
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def serialize(items):
            output = []
            for item in items:
                relative_path = os.path.relpath(item.impath, path_prefix)
                output.append((relative_path, item.label, item.classname))
            return output

        write_json(
            {
                "train": serialize(train),
                "val": serialize(val),
                "test": serialize(test),
            },
            filepath,
        )
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def deserialize(items):
            return [
                Datum(
                    impath=os.path.join(path_prefix, impath),
                    label=int(label),
                    classname=classname,
                )
                for impath, label, classname in items
            ]

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        return (
            deserialize(split["train"]),
            deserialize(split["val"]),
            deserialize(split["test"]),
        )

    @staticmethod
    def read_and_split_data(image_dir, p_train=0.5, p_val=0.2):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Dataset image directory not found: {image_dir}. "
                "See README.md for the expected layout."
            )

        categories = sorted(
            name
            for name in listdir_nohidden(image_dir)
            if os.path.isdir(os.path.join(image_dir, name))
        )
        train, val, test = [], [], []

        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = [
                os.path.join(category_dir, name)
                for name in listdir_nohidden(category_dir)
            ]
            random.shuffle(images)
            n_train = round(len(images) * p_train)
            n_val = round(len(images) * p_val)
            n_test = len(images) - n_train - n_val
            if min(n_train, n_val, n_test) <= 0:
                raise ValueError(
                    f"Class '{category}' needs samples in train, val and test splits"
                )

            def make_items(paths):
                return [
                    Datum(impath=path, label=label, classname=category)
                    for path in paths
                ]

            train.extend(make_items(images[:n_train]))
            val.extend(make_items(images[n_train : n_train + n_val]))
            test.extend(make_items(images[n_train + n_val :]))

        return train, val, test

    @staticmethod
    def subsample_classes(*splits, subsample="all"):
        if subsample not in {"all", "base", "new"}:
            raise ValueError("subsample must be one of: all, base, new")
        if subsample == "all":
            return splits

        labels = sorted({item.label for item in splits[0]})
        midpoint = math.ceil(len(labels) / 2)
        selected = labels[:midpoint] if subsample == "base" else labels[midpoint:]
        relabel = {old: new for new, old in enumerate(selected)}
        print(f"Subsampling {subsample} classes: {selected}")

        output = []
        for split in splits:
            output.append(
                [
                    Datum(
                        impath=item.impath,
                        label=relabel[item.label],
                        classname=item.classname,
                    )
                    for item in split
                    if item.label in relabel
                ]
            )
        return output


def _register_datasets():
    for name in SUPPORTED_DATASETS:
        dataset_class = type(
            name,
            (BiomedicalDataset,),
            {"dataset_dir": name, "__module__": __name__},
        )
        DATASET_REGISTRY.register(dataset_class)


_register_datasets()
