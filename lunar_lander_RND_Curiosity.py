import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import os

from rl_utils.SARST_RandomAccess_MemoryBuffer import SARST_RandomAccess_MemoryBuffer
from rl_utils.UncertaintyService import UncertaintyService

# enable XLA
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3"'

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')
state_space_shape = env.observation_space.shape[0]
action_space_shape = env.action_space.shape[0]

batch_size = 100
num_episodes = 5000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
predictor_learning_rate = 0.0001
gamma = 0.99
tau = 0.005
gradient_step = 1
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(env.action_space.shape)

uncertanicyService = UncertaintyService(state_space_shape, int(1.5 * state_space_shape), curiosity_mode=True)

extrinsic_reward_coef = tf.constant(1, dtype=tf.float32) #1
intrinsic_reward_coef = tf.constant(1, dtype=tf.float32) #10
actor_loss_coef = tf.constant(1, dtype=tf.float32) #1

initializer_bounds = 3e-3

RND_SEED = 0x12345

checkpoint_step = 500
max_epoch_steps = 1000

actor_checkpoint_file_name = 'll_sac_actor_checkpoint.h5'
critic_1_checkpoint_file_name = 'll_sac_critic1_checkpoint.h5'
critic_2_checkpoint_file_name = 'll_sac_critic2_checkpoint.h5'

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

def policy_network(state_space_shape, action_space_shape):
    input = keras.layers.Input(shape=(state_space_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    mean_output = keras.layers.Dense(action_space_shape, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(action_space_shape, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=[mean_output, log_std_dev_output])
    return model

def critic_network(state_space_shape, action_space_shape):
    input = keras.layers.Input(shape=(state_space_shape))
    actions_input = keras.layers.Input(shape=(action_space_shape))

    x = keras.layers.Concatenate()([input, actions_input])
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model
'''
SAC uses action reparametrization to avoid expectation over action.
So action is represented by squashed (tanh in this case) Normal distribution
'''
@tf.function
def get_actions(mu, log_sigma, noise=None):
    if noise is None:
        noise = gaus_distr.sample()
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

@tf.function
def get_log_probs(mu, sigma, actions):
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    z = gaus_distr.sample()
    # appendix C of the SAC paper discribe applyed boundings which is log(1-tanh(u)^2)
    log_probs = action_distributions.log_prob(mu + sigma*z) - \
                tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + action_bounds_epsilon), axis=1)
    return log_probs

@tf.function
def train_critics(states, actions, next_states, rewards, dones):
    mu, log_sigma = actor(next_states)
    mu = tf.squeeze(mu)
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

    target_actions = get_actions(mu, log_sigma)

    min_q = tf.math.minimum(target_critic_1([next_states, target_actions], training=False), \
                            target_critic_2([next_states, target_actions], training=False))
    min_q = tf.squeeze(min_q, axis=1)

    sigma = tf.math.exp(log_sigma)
    log_probs = get_log_probs(mu, sigma, target_actions)
    next_values = min_q - tf.math.exp(alpha_log) * log_probs # min(Q1^,Q2^) - alpha * logPi

    target_q = rewards + gamma * (1 - dones) * next_values

    with tf.GradientTape() as tape:
        current_q = critic_1([states, actions], training=True)
        c1_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c1_loss, critic_1.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_1.trainable_variables))

    with tf.GradientTape() as tape:
        current_q = critic_2([states, actions], training=True)
        c2_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c2_loss, critic_2.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_2.trainable_variables))
    return c1_loss, c2_loss

@tf.function
def train_actor(states):
    alpha = tf.math.exp(alpha_log)
    with tf.GradientTape() as tape:
        mu, log_sigma = actor(states, training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma)
        
        target_q = tf.math.minimum(critic_1([states, target_actions], training=False), \
                                   critic_2([states, target_actions], training=False))
        target_q = tf.squeeze(target_q, axis=1)
        
        sigma = tf.math.exp(log_sigma)
        log_probs = get_log_probs(mu, sigma, target_actions)

        actor_loss = tf.reduce_mean(alpha * log_probs - target_q) * actor_loss_coef
        
        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(alpha_log * tf.stop_gradient(log_probs + target_entropy))
        alpha_gradients = alpha_tape.gradient(alpha_loss, alpha_log)
        alpha_optimizer.apply_gradients([(alpha_gradients, alpha_log)])
    gradients = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return actor_loss

def soft_update_models():
    target_critic_1_weights = target_critic_1.get_weights()
    critic_1_weights = critic_1.get_weights()
    updated_critic_1_weights = []
    for cw,tcw in zip(critic_1_weights, target_critic_1_weights):
        updated_critic_1_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic_1.set_weights(updated_critic_1_weights)

    target_critic_2_weights = target_critic_2.get_weights()
    critic_2_weights = critic_2.get_weights()
    updated_critic_2_weights = []
    for cw,tcw in zip(critic_2_weights, target_critic_2_weights):
        updated_critic_2_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic_2.set_weights(updated_critic_2_weights)

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    actor = policy_network(state_space_shape, action_space_shape)
    print("New model created.")

if os.path.isfile(critic_1_checkpoint_file_name):
    critic_1 = keras.models.load_model(critic_1_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic_1 = critic_network(state_space_shape, action_space_shape)
    print("New Critic model created.")
target_critic_1 = critic_network(state_space_shape, action_space_shape)
target_critic_1.set_weights(critic_1.get_weights())

if os.path.isfile(critic_2_checkpoint_file_name):
    critic_2 = keras.models.load_model(critic_2_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic_2 = critic_network(state_space_shape, action_space_shape)
    print("New Critic model created.")
target_critic_2 = critic_network(state_space_shape, action_space_shape)
target_critic_2.set_weights(critic_2.get_weights())

rewards_history = []
global_step = 0

for i in range(num_episodes):
    done = False
    observation = env.reset()

    episodic_reward = 0
    intrinsic_total = 0
    intrinsic_count = 1
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

        if global_step > 4 * batch_size:
            states, actions, next_states, rewards, dones = exp_buffer(batch_size)

            intrinsic_rewards = uncertanicyService.getUncertainty(states)
            combined_rewards = extrinsic_reward_coef * rewards + intrinsic_reward_coef * intrinsic_rewards

            intrinsic_total += tf.math.reduce_sum(intrinsic_rewards)
            intrinsic_count += batch_size

            for _ in range(gradient_step):
                critic1_loss, critic2_loss = train_critics(states, actions, next_states, combined_rewards, dones)
                critic_loss_history.append(critic1_loss)
                critic_loss_history.append(critic2_loss)
            
                actor_loss = train_actor(states)
                actor_loss_history.append(actor_loss)
            soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    if i % checkpoint_step == 0 and i > 0:
        actor.save(actor_checkpoint_file_name)
        critic_1.save(critic_1_checkpoint_file_name)
        critic_2.save(critic_2_checkpoint_file_name)

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} TotalE: {episodic_reward:.4f} AvgE: {episodic_reward/epoch_steps:.4f} TotalI: {intrinsic_total:.4f} AvgI: {intrinsic_total/intrinsic_count:.4f} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_sac_curiosity.h5')
env.close()
input("training complete...")