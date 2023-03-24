import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import os
from rl_utils.SARST_RandomAccess_MemoryBuffer import SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')
X_shape = (env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 100
num_episodes = 5000
actor_learning_rate = 5e-5
critic_learning_rate = 8e-5
alpha_learning_rate = 5e-5
gamma = 0.99
tau = 0.005
gradient_step = 1
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(env.action_space.shape)

initializer_bounds = 3e-3

RND_SEED = 0x12345

max_epoch_steps = 1000
global_step = 0
update_interval_m = 2

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=[mean_output, log_std_dev_output])
    return model

def critic_network():
    input = keras.layers.Input(shape=(X_shape))
    actions_input = keras.layers.Input(shape=(outputs_count))

    x = keras.layers.Concatenate()([input, actions_input])
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    mean_output = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=[mean_output, log_std_dev_output])
    return model

@tf.function
def get_actions(mu, log_sigma, noise=None):
    if noise is None:
        noise = gaus_distr.sample()
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

@tf.function
def get_Q(q, q_sigma_log, noise=None):
    if noise is None:
        noise = gaus_distr.sample()
    return q + tf.math.exp(q_sigma_log) * noise

@tf.function
def get_actions_log_loglikelihood(mu, log_sigma, target_actions, noise=None):
    sigma = tf.math.exp(log_sigma)
    if noise is None:
        noise = gaus_distr.sample()
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    log_like = action_distributions.log_prob(mu + sigma * noise) - tf.reduce_mean(tf.math.log(1 - tf.math.pow(target_actions, 2) + action_bounds_epsilon), axis=1)
    return log_like

@tf.function
def train_critics(states, actions, next_states, rewards, dones):
    mu, log_sigma = target_actor(next_states, training=False)
    mu = tf.squeeze(mu)
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)
    target_actions = get_actions(mu, log_sigma)

    target_q_mean, target_q_log_sigma = target_critic([next_states, target_actions], training=False)
    target_q_mean = tf.squeeze(target_q_mean)
    target_q_log_sigma = tf.clip_by_value(tf.squeeze(target_q_log_sigma), log_std_min, log_std_max)
    q_next = get_Q(target_q_mean, target_q_log_sigma)

    log_likelihood = get_actions_log_loglikelihood(mu, log_sigma, target_actions)

    target_q = rewards + gamma * (1 - dones) * (q_next - tf.math.exp(alpha_log) * log_likelihood)

    with tf.GradientTape() as tape:
        q_mean, q_log_sigma = critic([states, actions], training=True)
        q_mean = tf.squeeze(q_mean)
        q_log_sigma = tf.clip_by_value(tf.squeeze(q_log_sigma), log_std_min, log_std_max)

        critic_loss = get_plain_critic_error(q_mean, tf.math.exp(q_log_sigma), target_q)
        # critic_loss = get_capped_critic_error(q_mean, tf.math.exp(q_log_sigma), target_q)
    gradients = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return critic_loss

@tf.function
def get_plain_critic_error(q_mean, q_sigma, target_q):
    q_distribution = tfp.distributions.Normal(loc=q_mean, scale=q_sigma)
    critic_loss = tf.reduce_mean(-q_distribution.log_prob(target_q)) # -log_likelihood(target_q | (q_mu,q_sigma))
    return critic_loss

@tf.function
def get_capped_critic_error(q_mean, q_sigma, target_q):
    bound_target_q = tf.clip_by_value(target_q, q_mean - 20, q_mean + 20)
    partial_dl_dQ =  tf.math.pow(q_mean - target_q, 2) / (2*tf.math.pow(tf.stop_gradient(q_sigma), 2))
    partial_dl_dSigma = tf.math.pow(tf.stop_gradient(q_mean) - bound_target_q, 2) / (2*tf.math.pow(q_sigma, 2)) + tf.math.log(q_sigma)
    return tf.reduce_mean(partial_dl_dQ + partial_dl_dSigma) # log likelihood

@tf.function
def train_actor(states):
    alpha = tf.math.exp(alpha_log)
    with tf.GradientTape() as tape:
        mu, log_sigma = actor(states, training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma)
        
        q_mean, q_log_sigma = critic([states, target_actions], training=False)
        q_mean = tf.squeeze(q_mean)
        q_log_sigma = tf.clip_by_value(tf.squeeze(q_log_sigma), log_std_min, log_std_max)
        target_q = get_Q(q_mean, q_log_sigma)
        
        log_likelihood = get_actions_log_loglikelihood(mu, log_sigma, target_actions)

        actor_loss = tf.reduce_mean(alpha * log_likelihood - target_q)
        
        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(alpha_log * tf.stop_gradient(log_likelihood + target_entropy))
        alpha_gradients = alpha_tape.gradient(alpha_loss, alpha_log)
        alpha_optimizer.apply_gradients([(alpha_gradients, alpha_log)])

    gradients = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return actor_loss

def soft_update_models():
    target_critic_weights = target_critic.get_weights()
    critic_weights = critic.get_weights()
    updated_critic_weights = []
    for cw,tcw in zip(critic_weights, target_critic_weights):
        updated_critic_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic.set_weights(updated_critic_weights)

    target_actor_weights = target_actor.get_weights()
    actor_weights = actor.get_weights()
    updated_actor_weights = []
    for cw,tcw in zip(actor_weights, target_actor_weights):
        updated_actor_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_actor.set_weights(updated_actor_weights)

actor = policy_network()
target_actor = policy_network()
target_actor.set_weights(actor.get_weights())

critic = critic_network()
target_critic = critic_network()
target_critic.set_weights(critic.get_weights())

rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []

    while not done:
        #env.render()
        mean, log_std_dev = actor(np.expand_dims(observation, axis = 0), training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])

        next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])

        exp_buffer.store(observation, [throttle_action, eng_ctrl_action], next_observation, reward, float(done))

        if global_step > 10 * batch_size:
            states, actions, next_states, rewards, dones = exp_buffer(batch_size)

            critic_loss = train_critics(states, actions, next_states, rewards, dones)
            critic_loss_history.append(critic_loss)
            
            if global_step % update_interval_m == 0:
                actor_loss = train_actor(states)
                actor_loss_history.append(actor_loss)
                soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_dsac.h5')
env.close()
input("training complete...")
