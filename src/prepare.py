import os

import numpy as np
import tensorflow as tf

OUT_DIR = "data/raw"


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), images=x_train, labels=y_train)
    np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), images=x_test, labels=y_test)
    print(f"Saved {len(x_train)} train and {len(x_test)} test images to {OUT_DIR}")


if __name__ == "__main__":
    main()
