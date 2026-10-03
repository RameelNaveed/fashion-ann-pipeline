import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
PIXEL_MEAN = 0.2860
PIXEL_STD = 0.3530


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["preprocess"]


def normalize(images):
    images = images.astype("float32") / 255.0
    return (images - PIXEL_MEAN) / PIXEL_STD


def main():
    params = load_params()
    train = np.load(os.path.join(RAW_DIR, "train.npz"))
    test = np.load(os.path.join(RAW_DIR, "test.npz"))

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["images"]),
        train["labels"],
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=train["labels"],
    )
    x_test = normalize(test["images"])

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), images=x_train, labels=y_train)
    np.savez_compressed(os.path.join(OUT_DIR, "val.npz"), images=x_val, labels=y_val)
    np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), images=x_test, labels=test["labels"])
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


if __name__ == "__main__":
    main()
