# https://arxiv.org/abs/2201.00042
import tensorflow as tf
from tensorflow.keras.layers import InputSpec

'''
Lunar lander:
* abs max - 1880
* max - 2000
'''
class ADLayer_Impl_2(tf.keras.layers.Dense):
    def __init__(self, units, 
                    dendrits_count,
                    context_vector_length,
                    use_abs_max = False,
                    **kwargs) -> None:
        super(ADLayer_Impl_2, self).__init__(units, **kwargs)
        self.neurons_count = units
        self.dendrit_segments_count = dendrits_count
        self.context_vector_length = context_vector_length
        self.use_abs_max = use_abs_max
        # important to override Dense layer's input spec
        self.input_spec = [InputSpec(dtype=tf.float32, ndim=2), InputSpec(dtype=tf.float32, shape=(None, self.context_vector_length))]

    def build(self, input_shape):
        super(ADLayer_Impl_2, self).build(input_shape[0])
        # must be set here also and be more specific
        self.input_spec = [InputSpec(dtype=tf.float32, shape=(None, input_shape[0][-1])),
                           InputSpec(dtype=tf.float32, shape=(None, self.context_vector_length))]
        self.dendrits = self.add_weight("dendrits", 
                                        shape=[self.context_vector_length, self.dendrit_segments_count * self.neurons_count],
                                        trainable=True, 
                                        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1))

    def call(self, inputs):
        context_vectors_batch = inputs[1]
        dense = super(ADLayer_Impl_2, self).call(inputs[0])
        batch_size = tf.shape(context_vectors_batch)[0]

        dendrits = context_vectors_batch @ self.dendrits # (batch_size, dendrits_count * neurons_count)
        dendrits = tf.reshape(dendrits, shape=(batch_size, self.dendrit_segments_count, self.neurons_count)) # (batch_size, dendrits_count, neurons_count)

        if not self.use_abs_max: # chapter 3.1 approach
            dendrit_idx = tf.argmax(tf.reduce_sum(dendrits, axis=-1), axis=-1) # (batch_size,)
        else: # chapter 6.3 approach
            dendrit_idx = tf.argmax(tf.reduce_sum(tf.math.abs(dendrits), axis=-1), axis=-1) # (batch_size,)
        active_dendrits = tf.gather(dendrits,dendrit_idx, axis=1, batch_dims=1)
        return dense * tf.math.sigmoid(active_dendrits)

    def get_config(self):
        config = super(ADLayer_Impl_2, self).get_config()
        config["dendrit_segments_count"] = self.dendrits_count
        config["context_vector_length"] = self.context_vector_length
        config["use_abs_max"] = self.use_abs_max
        return config

class ADLayer(tf.keras.layers.Dense):
    def __init__(self, units, 
                    dendrits_count,
                    context_vector_length,
                    use_abs_max = False,
                    **kwargs) -> None:
        super(ADLayer, self).__init__(units, **kwargs)
        self.neurons_count = units
        self.dendrit_segments_count = dendrits_count
        self.context_vector_length = context_vector_length
        self.use_abs_max = use_abs_max
        # important to override Dense layer's input spec
        self.input_spec = [InputSpec(dtype=tf.float32, ndim=2), InputSpec(dtype=tf.float32, shape=(None, self.context_vector_length))]

    def build(self, input_shape):
        super(ADLayer, self).build(input_shape[0])
        # must be set here also and be more specific
        self.input_spec = [InputSpec(dtype=tf.float32, shape=(None, input_shape[0][-1])),
                           InputSpec(dtype=tf.float32, shape=(None, self.context_vector_length))]
        self.dendrits = self.add_weight("dendrits", 
                                        shape=[self.context_vector_length, self.dendrit_segments_count * self.neurons_count],
                                        trainable=True, 
                                        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1))

    def call(self, inputs):
        context_vectors_batch = inputs[1]
        dense = super(ADLayer, self).call(inputs[0])
        batch_size = tf.shape(context_vectors_batch)[0]

        dendrits = context_vectors_batch @ self.dendrits # (batch_size, dendrits_count*neurons_count)
        dendrits = tf.reshape(dendrits, shape=(batch_size, self.neurons_count, self.dendrit_segments_count))

        if not self.use_abs_max: # chapter 3.1 approach
            active_dendrits = tf.reduce_max(dendrits, axis=-1) # (batch_size, neurons_count)
        else: # chapter 6.3 approach
            dendrit_idx = tf.argmax(tf.math.abs(dendrits), axis=-1) # (batch_size, self.neurons_count)
            selection_mask = tf.one_hot(dendrit_idx, depth=self.dendrit_segments_count, axis=-1) # 1 for dendrit_idx 0 otherwise
            active_dendrits = tf.reduce_sum(dendrits * selection_mask, axis=-1) # (batch_size, neurons_count)
        return dense * tf.math.sigmoid(active_dendrits)

    def get_config(self):
        config = super(ADLayer, self).get_config()
        config["dendrit_segments_count"] = self.dendrit_segments_count
        config["context_vector_length"] = self.context_vector_length
        config["use_abs_max"] = self.use_abs_max
        return config        

class kWTA_Layer(tf.keras.layers.Layer):
    def __init__(self, top_activations_count, **kwargs) -> None:
        super(kWTA_Layer, self).__init__(**kwargs)
        self.k = top_activations_count

    def call(self, inputs):
        threshold = tf.math.top_k(inputs, self.k).values[:,-1] # take top K and then last as threshold
        return tf.where(tf.math.less(inputs, tf.expand_dims(threshold, axis=-1)), tf.zeros_like(inputs), inputs)

    def get_config(self):
        config = super(kWTA_Layer, self).get_config()
        config["top_activations_count"] = self.k
        return config