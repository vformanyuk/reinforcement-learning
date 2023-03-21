import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl_utils.SARST_NStepReturn_RandomAccess_MemoryBuffer import SARST_NStepReturn_RandomAccess_MemoryBuffer

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 10000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
X_shape = (env.observation_space.shape[0])
gamma = 0.99
entropy_beta = 0.01
N_return = 3
batch_size = 16
log_std_min=-20
log_std_max=2

outputs_count = env.action_space.n

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

rewards_history = []

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

exp_buffer = SARST_NStepReturn_RandomAccess_MemoryBuffer(1001, N_return, gamma, env.observation_space.shape, env.action_space.shape, np.int32)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu', kernel_initializer = keras.initializers.HeNormal(seed = RND_SEED),
                                                   bias_initializer = keras.initializers.Constant(0.0))(input)
    x = keras.layers.Dense(256, activation='relu', kernel_initializer = keras.initializers.HeNormal(seed = RND_SEED),
                                                   bias_initializer = keras.initializers.Constant(0.0003))(x)
    actions_layer = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=actions_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu', kernel_initializer = keras.initializers.HeNormal(seed = RND_SEED),
                                                   bias_initializer = keras.initializers.Constant(0.0))(input)
    x = keras.layers.Dense(256, activation='relu', kernel_initializer = keras.initializers.HeNormal(seed = RND_SEED),
                                                   bias_initializer = keras.initializers.Constant(0.0003))(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

actor = policy_network()
critic = value_network()

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
        loss = - tf.reduce_mean(tf.reduce_sum(actions_log_distribution * one_hot_actions_mask, axis=1) * advantages) + entropy_beta * entropy
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
    exp_buffer.reset()

    while not done:
        actions_logits = actor(np.expand_dims(observation, axis = 0), training=False)
        actions_distribution = tf.nn.softmax(actions_logits)[0].numpy()
        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        
        next_observation, reward, done, _ = env.step(chosen_action)
        episodic_reward += reward
        
        exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

        if (epoch_steps % (batch_size + N_return) == 0 and epoch_steps > 0) or done:
            if done:
                states, actions, next_states, rewards, gammas, dones = exp_buffer.get_tail_batch(batch_size)
            else:
                states, actions, next_states, rewards, gammas, dones = exp_buffer(batch_size)
            critic_loss, adv = train_critic(states, next_states, rewards, gammas, dones)
            critic_loss_history.append(critic_loss)
            actor_loss = train_actor(states, actions, adv)
            actor_loss_history.append(actor_loss)

        epoch_steps+=1
        observation = next_observation

    rewards_history.append(episodic_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor mloss: {np.mean(actor_loss_history):.4f} Critic mloss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
if last_mean > 200:
    actor.save('lunar_lander_a2c_nrH.h5')
input("training complete...")