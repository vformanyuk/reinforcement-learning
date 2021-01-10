import tensorflow as tf
from tensorflow import keras

RND_SEED = 0x12345

def policy_network(input_shape, outputs_count):
    input = keras.layers.Input(shape=(input_shape))
    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(x)
    #x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(outputs_count, activation='tanh',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def critic_network(input_shape, outputs_count):
    actions_input = keras.layers.Input(shape=(outputs_count))
    input = keras.layers.Input(shape=(input_shape))

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Concatenate()([x, actions_input])
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.BatchNormalization()(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                kernel_regularizer = keras.regularizers.l2(0.01),
                                bias_regularizer = keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

def q_network(input_shape, outputs_count):
    input = keras.layers.Input(shape=(input_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    output = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model