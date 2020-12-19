import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl_utils.SARST_NStepReturn_RandomAccess_MemoryBuffer import SARST_NStepReturn_RandomAccess_MemoryBuffer
import os

'''
Try also:
Lambda returns. G(t) = R(t+1) + gamma*(1-lambda(t+1))*V(S[t+1]) + gamma * lambda(t+1)*G(t+1)
'''

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 10000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
X_shape = (env.observation_space.shape[0])
gamma = 0.99
entropy_beta = 0.01
N_return = 3
log_std_min=-20
log_std_max=2

checkpoint_step = 500

outputs_count = env.action_space.n

actor_checkpoint_file_name = 'll_a2c_nrH_checkpoint.h5'
critic_checkpoint_file_name = 'll_a2c_nrH_checkpoint.h5'

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

rewards_history = []

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

exp_buffer = SARST_NStepReturn_RandomAccess_MemoryBuffer(1001, N_return, gamma, env.observation_space.shape, env.action_space.shape, np.int32)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    actions_layer = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=actions_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name)
    print("Actor model restored from checkpoint.")
else:
    actor = policy_network()
    print("New Actor model created.")

if os.path.isfile(critic_checkpoint_file_name):
    critic = keras.models.load_model(critic_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic = value_network()
    print("New Critic model created.")

@tf.function
def train_actor(states, actions, advantages):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32)
    with tf.GradientTape() as tape:
        actions_logits = actor(states, training=True)
        actions_logits = tf.clip_by_value(actions_logits, log_std_min, log_std_max)
        actions_log_distribution = tf.nn.log_softmax(actions_logits)
        
        actions_distribution = tf.nn.softmax(actions_logits)
        entropy = -tf.reduce_sum(actions_log_distribution * actions_distribution)

        #loss = - actions_log_distribution[action] * advantages + entropy_beta * entropy
        loss = - tf.reduce_mean(tf.reduce_sum(tf.math.multiply(actions_log_distribution, one_hot_actions_mask), axis=1) * advantages) + entropy_beta * entropy
    gradients = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return loss

@tf.function
def train_critic(states, next_states, rewards, gamma_powers, dones):
    with tf.GradientTape() as tape:
        current_state_value = tf.squeeze(critic(states, training=True))
        n_step_return = rewards + tf.math.pow(gamma, gamma_powers + 1) * tf.squeeze(critic(next_states, training=False)) * (1 - dones)
        
        advantage = n_step_return - current_state_value # TD(N) error
        loss = tf.reduce_mean(tf.math.pow(advantage, 2)) #mse
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, advantage

for i in range(num_episodes):
    observation = env.reset()
    epoch_steps = 0
    done = False
    episodic_reward = 0
    critic_loss_history = []
    actor_loss_history = []

    while not done:
        actions_logits = actor(np.expand_dims(observation, axis = 0), training=False)
        actions_distribution = tf.nn.softmax(actions_logits)[0].numpy()
        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        
        next_observation, reward, done, _ = env.step(chosen_action)
        episodic_reward += reward
        
        exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

        if exp_buffer.is_buffer_ready()[0] and epoch_steps % N_return == 0:
            states, actions, next_states, rewards, gammas, dones = exp_buffer.get_last_batch() # batch_size = N, instead of random sampling use 'sliding window'
            critic_loss, adv = train_critic(states, next_states, rewards, gammas, dones)
            critic_loss_history.append(critic_loss)
            actor_loss = train_actor(states, actions, adv)
            actor_loss_history.append(actor_loss)

        epoch_steps+=1
        observation = next_observation

    if i % checkpoint_step == 0 and i > 0:
        actor.save(actor_checkpoint_file_name)
        critic.save(critic_checkpoint_file_name)

    rewards_history.append(episodic_reward)

    exp_buffer.reset()

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor mloss: {np.mean(actor_loss_history):.4f} Critic mloss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
if last_mean > 200:
    actor.save('lunar_lander_a2c_nrH.h5')
input("training complete...")