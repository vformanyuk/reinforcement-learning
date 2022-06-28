import tensorflow as tf
from tensorflow import keras

RND_SEED = 0x12345
SAC_INITIALIZER_BOUNDS = 3e-3

def policy_network(state_space_shape, outputs_count, actor_recurrent_layer_size):
    input = keras.layers.Input(shape=(state_space_shape))
    hidden_input = keras.layers.Input(shape=(actor_recurrent_layer_size))

    rnn_out, hx = keras.layers.GRU(actor_recurrent_layer_size, return_state=True)(input, initial_state=[hidden_input])
    x = keras.layers.Dense(256, activation='relu')(rnn_out)
    x = keras.layers.Dense(128, activation='relu')(x)
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED))(x)
    log_sigma_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, hidden_input], outputs=[mean_output, log_sigma_output, hx])
    return model

def critic_network(state_space_shape, outputs_count, actor_recurrent_layer_size):
    input = keras.layers.Input(shape=(state_space_shape))
    actions_input = keras.layers.Input(shape=(outputs_count))
    hidden_input = keras.layers.Input(shape=(actor_recurrent_layer_size))

    rnn_out, hx = keras.layers.GRU(actor_recurrent_layer_size, return_state=True)(input, initial_state=[hidden_input])
    x = keras.layers.Concatenate()([rnn_out, actions_input])

    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input, hidden_input], outputs=[q_layer, hx])
    return model

def value_network(state_space_shape):
    state_input = keras.layers.Input(shape=(state_space_shape))

    x = keras.layers.Flatten()(state_input)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    v_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-SAC_INITIALIZER_BOUNDS, maxval=SAC_INITIALIZER_BOUNDS, seed=RND_SEED))(x)

    model = keras.Model(inputs=state_input, outputs=v_layer)
    return model
