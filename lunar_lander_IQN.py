import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl_utils.SARST_RandomAccess_MemoryBuffer import \
    SARST_RandomAccess_MemoryBuffer
from rl_utils.cos_layer import CosLayer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 2500
learning_rate = 3e-4
batch_size = 128
state_space_shape = env.observation_space.shape[0]
action_space_shape = env.action_space.n
gamma = 0.99

global_step = 0
copy_step = 50
steps_train = 4
start_steps = 2000

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 1.5e-4
tau = 0.005

cos_layer_size = 64
quantile_count_K = 16
kappa = tf.constant(1.0, dtype=tf.float32)

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

optimizer = tf.keras.optimizers.Adam(learning_rate)

def q_network_no_embedding(state_space_shape, action_space_shape, cos_layer_size, quantiles_count):
    input = keras.layers.Input(shape=state_space_shape, batch_size=batch_size)
    tau_input = keras.layers.Input(shape=(quantiles_count,), batch_size=batch_size)

    x = keras.layers.Dense(cos_layer_size * 4, activation='relu')(input) # 256
    x = keras.layers.Dense(cos_layer_size * 2, activation='relu')(x) # 128
    x = keras.layers.Dense(cos_layer_size, activation='relu')(x) # 64

    cos_x = CosLayer(cos_layer_size)(tau_input) # 64
    cos_x = tf.keras.layers.ReLU()(cos_x)
    
    x = tf.keras.layers.Multiply()([x, cos_x])
    output = keras.layers.Dense(action_space_shape * quantiles_count, activation='linear')(x)

    model = keras.Model(inputs=[input, tau_input], outputs=output)
    return model

def q_network_embedding(state_space_shape, action_space_shape, cos_layer_size, quantiles_count):
    input = keras.layers.Input(shape=state_space_shape, batch_size=batch_size)
    tau_input = keras.layers.Input(shape=(quantiles_count,), batch_size=batch_size)

    x = keras.layers.Dense(256, activation='relu')(input)
    
    cos_x = CosLayer(cos_layer_size)(tau_input)
    cos_x_embeding = keras.layers.Dense(256, activation='relu')(cos_x)
    
    x = tf.keras.layers.Multiply()([x, cos_x_embeding])
    x = keras.layers.Dense(128, activation='relu')(x)
    output = keras.layers.Dense(action_space_shape * quantiles_count, activation='linear')(x)

    model = keras.Model(inputs=[input, tau_input], outputs=output)
    return model

def epsilon_greedy(observation, epsilon_threshold):
    if np.random.rand() < epsilon_threshold:
        return np.random.randint(action_space_shape)
    else:
        inference_tau = tf.random.uniform(shape=(1, quantile_count_K))
        q_value = mainQ([np.expand_dims(observation, axis = 0), inference_tau])
        q_value = np.reshape(q_value, newshape=(action_space_shape, quantile_count_K))
        return np.argmax(np.mean(q_value, axis=1))

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

@tf.function
def learn(states, actions, next_states, rewards, dones):
    quantile_tau_next = tf.random.uniform(shape=(batch_size, quantile_count_K))
    quantile_tau = tf.random.uniform(shape=(batch_size, quantile_count_K))

    next_q = targetQ([next_states, quantile_tau_next], training=False)
    next_q = tf.reshape(next_q, shape=(batch_size, action_space_shape, quantile_count_K))
    next_actions = tf.math.argmax(tf.reduce_mean(next_q, axis=-1, keepdims=True), axis=1, output_type=tf.int32)
    next_q_masked = tf.gather(next_q, next_actions, axis=1, batch_dims=1) # (batch_size, 1, quantile_count_K)

    broadcasted_reward = tf.expand_dims(tf.expand_dims(rewards, axis=-1), axis=-1) # (batch_size,) => (batch_size, 1, 1)
    broadcasted_dones = tf.expand_dims(tf.expand_dims((1 - dones), axis=-1), axis=-1)
    target_q = broadcasted_reward + gamma * next_q_masked * broadcasted_dones # (batch_size, 1, quantile_count_K)

    with tf.GradientTape() as tape:
        current_q = mainQ([states, quantile_tau], training=True) # (batch_size, action_space_shape)
        current_q = tf.reshape(current_q, shape=(batch_size, action_space_shape, quantile_count_K))
        current_q = tf.gather(current_q, tf.expand_dims(actions, axis=-1), axis=1, batch_dims=1) # (batch_size, 1, quantile_count_K)
        current_q = tf.transpose(current_q, [0,2,1]) # (batch_size, quantile_count_K, 1)

        td_error = target_q - current_q # (batch_size, quantile_count_K, quantile_count_K)
        h_loss = hubber_loss(tf.abs(td_error))
        diracs = tf.where(tf.math.less(td_error, 0.0), 1.0, 0.0)
        loss = tf.abs(tf.expand_dims(quantile_tau, axis=-1) - diracs) * h_loss
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=1)
    gradients = tape.gradient(loss, mainQ.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mainQ.trainable_variables))
    return tf.reduce_mean(loss)

@tf.function
def hubber_loss(abs_td_error):
    return tf.where(tf.math.less(abs_td_error, kappa), 0.5 * tf.math.pow(abs_td_error, 2), kappa * (abs_td_error - 0.5 * kappa))

exp_buffer = SARST_RandomAccess_MemoryBuffer(500_000, (state_space_shape,), None, action_type=np.int32)

mainQ = q_network_embedding(state_space_shape, action_space_shape, cos_layer_size, quantile_count_K)
targetQ = q_network_embedding(state_space_shape, action_space_shape, cos_layer_size, quantile_count_K)

rewards_history = []

for i in range(num_episodes):
    done = False
    obs = env.reset()
    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    while not done:
        #env.render()
        chosen_action = epsilon_greedy(obs, epsilon)
        next_obs, reward, done, _ = env.step(chosen_action)
        
        exp_buffer.store(tf.convert_to_tensor(obs, dtype=tf.float32),
                         tf.convert_to_tensor(chosen_action, dtype=tf.int32),
                         tf.convert_to_tensor(next_obs, dtype=tf.float32),
                         tf.convert_to_tensor(reward, dtype=tf.float32),
                         tf.convert_to_tensor(float(done), dtype=tf.float32))
        
        if global_step > start_steps:
            states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor = exp_buffer(batch_size)
            loss = learn(states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor)
            episodic_loss.append(loss)
            epsilon_decay()

        if (global_step + 1) % copy_step == 0 and global_step > start_steps:
            targetQ.set_weights(mainQ.get_weights())
        obs = next_obs
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward
    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Avg loss: {np.mean(episodic_loss):.4f} Epsilon: {epsilon:.4f} Total reward: {episodic_reward:.4f} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    targetQ.save('lunar_lander_IQN.h5')
env.close()
input("training complete...")