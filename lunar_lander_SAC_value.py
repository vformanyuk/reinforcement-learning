import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
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
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
value_learning_rate = 3e-4
gamma = 0.99
tau = 0.005
gradient_step = 1
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(env.action_space.shape)

initializer_bounds = 3e-3

RND_SEED = 0x12345

checkpoint_step = 500
max_epoch_steps = 1000
global_step = 0

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
value_optimizer = tf.keras.optimizers.Adam(value_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

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
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(X_shape))

    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    v_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

'''
SAC uses action reparametrization to avoid expectation over action.
So action is represented by squashed (tanh in this case) Normal distribution
'''
@tf.function
def get_actions(mu, log_sigma, noise):
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

@tf.function
def get_log_probs(mu, sigma, actions, noise):
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    log_probs = action_distributions.log_prob(mu + sigma * noise) - \
                tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + action_bounds_epsilon), axis=1)
    return log_probs

@tf.function
def train_critics(states, actions, next_states, rewards, dones):
    target_q = rewards + gamma * (1 - dones) * target_value_net(next_states, training=False)

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
def train_actor_and_value(states):
    noise = gaus_distr.sample(sample_shape=(batch_size, outputs_count))
    with tf.GradientTape() as tape:
        mu, log_sigma = actor(states, training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma, noise)
        log_probs = get_log_probs(mu, tf.math.exp(log_sigma), target_actions, noise)
        
        target_q = tf.math.minimum(critic_1([states, target_actions], training=False), \
                                   critic_2([states, target_actions], training=False))
        target_q = tf.squeeze(target_q, axis=1)
        target_v = tf.stop_gradient(target_q - log_probs)
        actor_loss = tf.reduce_mean(log_probs - target_q)

    gradients = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))

    with tf.GradientTape() as value_tape:
        current_v = value_net(states, training=True)
        value_loss = mse_loss(current_v, target_v)
    value_gradient = value_tape.gradient(value_loss, value_net.trainable_variables)
    value_optimizer.apply_gradients(zip(value_gradient, value_net.trainable_variables))

    return actor_loss, value_loss

def soft_update_models():
    target_value_weights = target_value_net.get_weights()
    value_weights = value_net.get_weights()
    updated_target_value_weights = []
    for vw,tvw in zip(value_weights, target_value_weights):
        updated_target_value_weights.append(tau * vw + (1.0 - tau) * tvw)
    target_value_net.set_weights(updated_target_value_weights)

actor = policy_network()
critic_1 = critic_network()
critic_2 = critic_network()
value_net = value_network()
target_value_net = value_network()
target_value_net.set_weights(value_net.get_weights())

rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []
    value_loss_history = []

    while not done:
        #env.render()
        mean, log_std_dev = actor(np.expand_dims(observation, axis = 0), training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0], gaus_distr.sample())
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1], gaus_distr.sample())

        next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])

        exp_buffer.store(observation, [throttle_action, eng_ctrl_action], next_observation, reward, float(done))

        if global_step > 5 * batch_size:
            states, actions, next_states, rewards, dones = exp_buffer(batch_size)

            for _ in range(gradient_step):
                critic1_loss, critic2_loss = train_critics(states, actions, next_states, rewards, dones)
                critic_loss_history.append(critic1_loss)
                critic_loss_history.append(critic2_loss)
            
                actor_loss, value_loss = train_actor_and_value(states)
                actor_loss_history.append(actor_loss)
                value_loss_history.append(value_loss)
            soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Value_Loss: {np.mean(value_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_sac_value.h5')
env.close()
input("training complete...")
