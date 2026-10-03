import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

DATA_DIR = "data/processed"
MODEL_PATH = "models/model.h5"
METRICS_PATH = "metrics.json"
PLOT_PATH = "confusion_matrix.png"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    test = np.load(os.path.join(DATA_DIR, "test.npz"))
    x_test, y_test = test["images"], test["labels"]

    model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(accuracy), 4)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=CLASS_NAMES, xticks_rotation=45, colorbar=False, ax=ax
    )
    ax.set_title("Fashion-MNIST Confusion Matrix")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"Test loss: {metrics['test_loss']}, Test accuracy: {metrics['test_accuracy']}")


if __name__ == "__main__":
    main()
