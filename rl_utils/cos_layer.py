import tensorflow as tf
import numpy as np

class CosLayer(tf.keras.layers.Dense):
    def __init__(self, units, **kwargs) -> None:
        super(CosLayer, self).__init__(units, **kwargs)
        self.pi_i = tf.expand_dims(tf.convert_to_tensor([i*np.pi for i in range(units)], dtype=tf.float32), axis=0) # (1, cos_layer)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1) * self.pi_i # (batch_size, tau_count, 1) @ (1, cos_layer) => (batch_size, tau_count, cos_layer)
        x = tf.math.cos(x)
        x = tf.reduce_sum(x, axis=-1) # (batch_size, tau_count, cos_layer) => (batch_size, tau_count)
        return x @ self.kernel + self.bias