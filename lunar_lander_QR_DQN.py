import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl_utils.SARST_RandomAccess_MemoryBuffer import \
    SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 800
learning_rate = 0.0005
batch_size = 128
state_shape = env.observation_space.shape[0]
action_space_shape = env.action_space.n
discount_factor = 0.99

global_step = 0
copy_step = 50
steps_train = 4
start_steps = 1000

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 1.5e-4
tau = 0.05

quantile_N = 32
kappa = tf.constant(1.0, dtype=tf.float32)

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

optimizer = tf.keras.optimizers.Adam(learning_rate)

quantile_tau = tf.convert_to_tensor([float(i)/quantile_N for i in range(1, quantile_N + 1)], dtype=tf.float32)
quantile_tau_tensor = tf.transpose(tf.tile(tf.expand_dims(quantile_tau, axis=-1), (1,batch_size))) # (batch_size, quantile_N)

def q_network():
    input = keras.layers.Input(shape=state_shape, batch_size=batch_size)
    x = keras.layers.Dense(512, activation='relu')(input)
    x = keras.layers.Dense(512, activation='relu')(x)
    output = keras.layers.Dense(action_space_shape * quantile_N, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def epsilon_greedy(observation, epsilon_threshold):
    if tf.random.uniform(shape=(1,))[0].numpy() < epsilon_threshold:
        return tf.random.uniform(shape=(1,), maxval=action_space_shape, dtype=tf.int32)[0]
    else:
        q_value = tf.squeeze(mainQ(tf.expand_dims(observation, axis = 0), training=False), axis=0)
        q_value = tf.reshape(q_value, shape=(action_space_shape, quantile_N))
        q_actions = tf.reduce_mean(q_value, axis=1) #reduce_sum
        return tf.math.argmax(q_actions, output_type=tf.int32)

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

def soft_update_models(q_network, target_q_network):
    target_q_weights = target_q_network.get_weights()
    q_weights = q_network.get_weights()
    updated_weights = []
    for cw,tcw in zip(q_weights, target_q_weights):
        updated_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_q_network.set_weights(updated_weights)

@tf.function
def learn(states, actions, next_states, rewards, dones):
    actions_mask = tf.one_hot(actions, depth=action_space_shape, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4
    actions_mask = tf.tile(tf.expand_dims(actions_mask, axis=-1), (1,1,quantile_N))
    
    next_q = targetQ(next_states, training=False)
    next_q = tf.reshape(next_q, shape=(batch_size, action_space_shape, quantile_N))
    next_actions = tf.math.argmax(tf.reduce_mean(next_q, axis=-1), axis=1, output_type=tf.int32)
    next_actions_mask = tf.one_hot(next_actions, action_space_shape, on_value=1.0, off_value=0.0)
    next_actions_mask = tf.tile(tf.expand_dims(next_actions_mask, axis=-1), (1,1,quantile_N))
    next_q = tf.reduce_sum(next_actions_mask * next_q, axis = 1) # (batch_size, quantile_N)

    #rewards = tf.expand_dims(rewards, axis=-1) # for broadcasting
    broadcasted_reward = tf.tile(tf.expand_dims(rewards, axis=-1), (1, quantile_N))
    broadcasted_dones = tf.tile(tf.expand_dims((1 - dones), axis=-1), (1, quantile_N))
    target_q = broadcasted_reward + discount_factor * next_q * broadcasted_dones # shape = (batch_size, quantile_N)
    
    with tf.GradientTape() as tape:
        current_q = mainQ(states, training=True) # shape = (batch_size,4 * quantile_N)
        current_q = tf.reshape(current_q, shape=(batch_size, action_space_shape, quantile_N))
        td_error = target_q - tf.reduce_sum(actions_mask * current_q, axis = 1) # (batch_size, quantile_N)
        loss = tf.abs(quantile_tau_tensor - tf.where(tf.math.less(td_error, 0.0), 1.0, 0.0))*hubber_loss(td_error)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, mainQ.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mainQ.trainable_variables))
    return tf.reduce_mean(loss)

@tf.function
def hubber_loss(td_error):
    return tf.where(tf.math.less(td_error, kappa), 0.5 * tf.math.pow(td_error,2), kappa*(tf.abs(td_error) - 0.5*kappa))

exp_buffer = SARST_RandomAccess_MemoryBuffer(500_000, (state_shape,), None, action_type=np.int32)

mainQ = q_network()
targetQ = q_network()

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
        next_obs, reward, done, _ = env.step(chosen_action.numpy())
        exp_buffer.store(tf.convert_to_tensor(obs, dtype=tf.float32),
                         tf.convert_to_tensor(chosen_action, dtype=tf.int32),
                         tf.convert_to_tensor(next_obs, dtype=tf.float32),
                         tf.convert_to_tensor(reward, dtype=tf.float32),
                         tf.convert_to_tensor(float(done), dtype=tf.float32))
        
        if global_step % steps_train == 0 and global_step > start_steps:
            states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor = exp_buffer(batch_size)
            loss = learn(states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor)
            episodic_loss.append(loss)
            epsilon_decay()
            soft_update_models(mainQ, targetQ)

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
    targetQ.save('lunar_doubleDQN.h5')
env.close()
input("training complete...")