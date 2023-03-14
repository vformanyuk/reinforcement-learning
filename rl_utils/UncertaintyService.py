import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class UncertaintyService:
    def __init__(self, state_shape, embedding_shape, curiosity_mode=False) -> None:
        self.states_moving_cma = tf.Variable(shape=(state_shape,), trainable=False, initial_value=tf.fill([state_shape,], 1e-6), dtype=tf.float32) # cumulitive moving avg
        self.states_moving_ssd = tf.Variable(shape=(state_shape,), trainable=False, initial_value=tf.fill([state_shape,], 1e-6), dtype=tf.float32) # sum of squared deviations
        self.embedding_moving_cma = tf.Variable(shape=(embedding_shape,), trainable=False, initial_value=tf.fill([embedding_shape,], 1e-6), dtype=tf.float32)
        self.embedding_moving_ssd = tf.Variable(shape=(embedding_shape,), trainable=False, initial_value=tf.fill([embedding_shape,], 1e-6), dtype=tf.float32)
        self.predictor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.steps = tf.Variable(trainable=False, initial_value=0.0, dtype=tf.float32)
        self.curiosity = curiosity_mode
        self.embedding = self._createModel(state_shape, embedding_shape)
        self.predictor = self._createModel(state_shape, embedding_shape)

    def _createModel(self, state_space_shape, embedding_shape):
        input = tf.keras.layers.Input(shape=(state_space_shape, ))
        x = tf.keras.layers.Dense(256, kernel_initializer = tf.keras.initializers.HeNormal(),
                                       bias_initializer = tf.keras.initializers.Zeros())(input)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(128, kernel_initializer = tf.keras.initializers.HeNormal(),
                                       bias_initializer = tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.LeakyReLU()(x)
        embedding_layer = tf.keras.layers.Dense(embedding_shape, activation='linear')(x)
        return tf.keras.Model(inputs=input, outputs=embedding_layer)
    
    @tf.function
    def getUncertainty(self, states):
        starting_step = self.steps.read_value()
        current_states_cma = self.states_moving_cma.read_value()
        states_cma = current_states_cma
        states_ssd = self.states_moving_ssd.read_value() # running sum of squared deviations
        for state in states:
            # Knuth, the art of computer programming vol2, p 232
            states_cma = states_cma + (state - states_cma)/(self.steps + 1)
            states_ssd += (state - current_states_cma)*(state - states_cma)
            current_states_cma = states_cma
            self.steps.assign_add(1)
        states_std_dev = tf.math.sqrt(states_ssd / (self.steps - 1)) # Knuth, the art of computer programming vol2, p 232
        
        self.states_moving_cma.assign(states_cma)
        self.states_moving_ssd.assign(states_ssd)

        normalized_states = (states - states_cma) / states_std_dev
        embedings = self.embedding(normalized_states, training=False)
        with tf.GradientTape() as tape:
            pred = self.predictor(normalized_states, training=True)
            uncertainty = tf.math.pow(pred - embedings, 2)
            uncertainty_loss = tf.reduce_mean(uncertainty)
        gradients = tape.gradient(uncertainty_loss, self.predictor.trainable_variables)
        self.predictor_optimizer.apply_gradients(zip(gradients, self.predictor.trainable_variables))

        if not self.curiosity:
            return uncertainty
        
        self.steps.assign(starting_step)
        current_curiosity_cma = self.embedding_moving_cma.read_value()
        curiosity_cma = current_curiosity_cma
        curiosity_ssd = self.embedding_moving_ssd.read_value()
        for u in uncertainty:
            # Knuth, the art of computer programming vol2, p 232
            curiosity_cma = curiosity_cma + (u - curiosity_cma)/(self.steps + 1)
            curiosity_ssd += (u - current_curiosity_cma)*(u - curiosity_cma)
            current_curiosity_cma = curiosity_cma
            self.steps.assign_add(1)
        curiosity_std_dev = tf.math.sqrt(curiosity_ssd / (self.steps - 1))

        self.embedding_moving_cma.assign(curiosity_cma)
        self.embedding_moving_ssd.assign(curiosity_ssd)
        return tf.reduce_mean((uncertainty - curiosity_cma) / curiosity_std_dev, axis=1) # normalize curiosity (intrinsic reward)