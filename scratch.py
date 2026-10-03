import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train[:10])
