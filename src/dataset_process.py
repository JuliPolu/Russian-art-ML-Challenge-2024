import pandas as pd

import os
import numpy as np

from typing import Tuple, List

from sklearn.model_selection import train_test_split

import torch
from PIL import Image


def load_image(img: str, transforms=None) -> torch.Tensor:
    img = Image.open(img).convert('RGB')
    if transforms is not None:
        img = transforms(img)
    return img


def dataset_prepare_split(
    train_csv_path: str, 
    train_data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    df = pd.read_csv(train_csv_path, sep="\t")
    labels = np.array(df["label_id"].tolist())
    files = np.array([os.path.join(train_data_path, fname) for fname in df["image_name"].tolist()])

    indices = list(range(len(labels)))
    ind_train, ind_test, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=139, stratify=labels)

    x_train, x_test = files[ind_train], files[ind_test]
    train_labels, test_labels = labels[ind_train], labels[ind_test]

    return x_train, train_labels, x_test, test_labels
