import tensorflow as tf
from tensorflow import keras

RND_SEED = 0x12345

def policy_network(input_shape, outputs_count):
    input = keras.layers.Input(shape=(input_shape))
    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(x)
    #x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(outputs_count, activation='tanh',
                                kernel_initializer = keras.initializers.VarianceScaling(scale=5/3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0))(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def critic_network(input_shape, outputs_count):
    actions_input = keras.layers.Input(shape=(outputs_count))
    input = keras.layers.Input(shape=(input_shape))

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           bias_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Concatenate()([x, actions_input])
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           bias_initializer = keras.initializers.HeUniform(seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.BatchNormalization()(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0),
                                kernel_regularizer = keras.regularizers.l2(0.01),
                                bias_regularizer = keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

SAC_INITIALIZER_BOUNDS = 3e-3

def sac_policy_network(input_shape, outputs_count):
    input = keras.layers.Input(shape=(input_shape))
    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0))(x)
    log_std_dev_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0))(x)

    model = keras.Model(inputs=input, outputs=[mean_output, log_std_dev_output])
    return model

def sac_critic_network(input_shape, outputs_count):
    input = keras.layers.Input(shape=(input_shape))
    actions_input = keras.layers.Input(shape=(outputs_count))

    x = keras.layers.Concatenate()([input, actions_input])
    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

def sac_value_network(input_shape):
    input = keras.layers.Input(shape=(input_shape))

    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(256, kernel_initializer = keras.initializers.HeUniform(seed=RND_SEED), bias_initializer = keras.initializers.HeUniform(seed=RND_SEED))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    v_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.Constant(value=0))(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model    