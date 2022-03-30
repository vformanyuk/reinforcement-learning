# https://arxiv.org/abs/2201.00042
import tensorflow as tf

class ADLayer(tf.keras.layers.Layer):
    def __init__(self, neurons_count, 
                    dendrits_count,
                    context_vector_length,
                    use_abs_max = False,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initalizer=tf.keras.initializers.GlorotNormal()) -> None:
        super().__init__()
        self.neurons_count = neurons_count
        self.dendrits_count = dendrits_count
        self.context_vector_length = context_vector_length
        self.kernel_initializer = kernel_initializer
        self.bias_initalizer = bias_initalizer
        self.use_abs_max = use_abs_max
        #activation ??

    def build(self, input_shape):
        nn_input_shape = input_shape[0]
        self.kernel = self.add_weight("kernel", shape=[nn_input_shape[-1], self.neurons_count], trainable=True, initializer = self.kernel_initializer)
        self.bias = self.add_weight("bias", shape=[self.neurons_count,], trainable=True, initializer = self.bias_initalizer)
        self.dendrits = self.add_weight("dendrits", 
                                        shape=[self.context_vector_length, self.dendrits_count, self.neurons_count],
                                        trainable=True, 
                                        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1))

    def call(self, inputs):
        nn_inputs_batch = inputs[0]
        context_vectors_batch = inputs[1]
        dense = nn_inputs_batch @ self.kernel + self.bias
        batch_size = tf.shape(context_vectors_batch)[0]

        dendrits = tf.reshape(self.dendrits, shape=(self.context_vector_length, self.dendrits_count*self.neurons_count)) #adjust dims for matmul
        dendrits = context_vectors_batch @ dendrits # (batch_size, dendrits_count * neurons_count)
        dendrits = tf.reshape(dendrits, shape=(batch_size, self.dendrits_count, self.neurons_count)) # restore dims

        if not self.use_abs_max: # chapter 3.1 approach
            active_dendrits = tf.reduce_max(dendrits, axis=1) # (batch_size, neurons_count)
        else: # chapter 6.3 approach
            dendrit_idx = tf.argmax(tf.math.abs(dendrits), axis=1)
            selection_mask = tf.one_hot(dendrit_idx, depth=self.dendrits_count, axis=1) # 1 for dendrit_idx 0 otherwise
            active_dendrits = tf.reduce_sum(dendrits * selection_mask, axis=1) # (batch_size, neurons_count)
        return dense * tf.math.sigmoid(active_dendrits)

class kWTA_Layer(tf.keras.layers.Layer):
    def __init__(self, top_activations_count) -> None:
        super().__init__()
        self.k = top_activations_count
    
    def call(self, inputs):
        threshold = tf.sort(inputs, direction='DESCENDING')[:self.k][-1] # sort, take top K, take last one
        return tf.where(tf.math.less(inputs, threshold), tf.zeros_like(inputs), inputs)