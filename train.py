import os

import numpy as np
import tensorflow as tf
import yaml

DATA_DIR = "data/processed"
MODEL_DIR = "models"


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]


def load_split(name):
    data = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return data["images"], data["labels"]


def build_model(params):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    params = load_params()
    tf.keras.utils.set_random_seed(params["seed"])
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_model(params)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
        callbacks=[tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "history.csv"))],
    )
    model.save(os.path.join(MODEL_DIR, "model.h5"))
    print(f"Saved model to {MODEL_DIR}/model.h5")


if __name__ == "__main__":
    main()
