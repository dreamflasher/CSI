#!/usr/bin/env python3
import copy
import os
import os.path as osp
import shutil
import sys
import tarfile
import urllib
from collections import defaultdict
from pathlib import Path

import requests
import torch
import torchvision
from PIL import Image
from scipy import io as scipy_io
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import (DatasetFolder, default_loader, has_file_allowed_extension,
                                         make_dataset)
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


class MVTAD(VisionDataset):

    URLS = {
        "images": (
            "ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz",
            "eefca59f2cede9c3fc5b6befbfec275e",
        ),
    }

    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
    )

    def __init__(
        self,
        root,
        normal_class,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    ):

        super(MVTAD, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = Path(root + "/mvtad").expanduser()
        os.makedirs(root + "/mvtad", exist_ok=True)
        self.dataset_name = "mvtad"
        if isinstance(normal_class, list):
            normal_class = normal_class[0]

        filename = "mvtec_anomaly_detection.tar.xz"
        if not (self.root / filename).exists():
            download_and_extract_archive(
                self.URLS["images"][0],
                self.root,
                filename=filename,
                md5=self.URLS["images"][1],
            )
        # hack dataset structure to conform OOD datasets by renaming "good" to "classname"
        # for dirpath, dirnames, filenames in os.walk(self.root):
        #   os.chmod(dirpath, 0o777)
        #   for filename in filenames:
        #     os.chmod(os.path.join(dirpath, filename), 0o666)
        # for p in self.root.iterdir():
        #   if p.is_dir():
        #     r = str(self.root / p)
        #     for t in ["train", "test"]:
        #       print(f"{r}/{t}/good", f"{r}/{t}/{p.name}")
        #       shutil.move(f"{r}/{t}/good", f"{r}/{t}/{p.name}")

        self.train = train

        # def is_valid_file(path: str):
        #   """
        #   The MVTAD dataset is structured as root/class_i/train and root/class_i/test, so we need to
        #   filter out paths according to the split
        #   """
        #   split = "train" if train else "test"
        #   if split in Path(path).parts and has_file_allowed_extension(
        #       path, self.IMG_EXTENSIONS):
        #     return True
        #   return False

        train_test = "train" if train else "test"
        class_labels = sorted(next(os.walk(self.root))[1])
        self.root = Path(self.root) / class_labels[normal_class] / train_test

        # super().__init__(str(folder),
        #                  loader=default_loader,
        #                  transform=transform,
        #                  target_transform=target_transform,
        #                  extensions=self.IMG_EXTENSIONS)
        extensions = self.IMG_EXTENSIONS

        classes, class_to_idx = self._find_classes(
            self.root, class_labels, normal_class
        )
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        print(len(samples))
        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir, class_labels, normal_class):
        """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.
        class_labels (list): additional labels to put in front
        normal_class (int): good will be reindexed to normal_class

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_labels.extend(classes)
        classes = class_labels
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_to_idx["good"] = normal_class
        return classes, class_to_idx

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    root = os.getenv("DATASETS_ROOT", "./")
    mvtad = MVTAD(root=root, train=True)
