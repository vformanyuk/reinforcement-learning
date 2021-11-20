import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from datetime import datetime
from tensorflow import keras
from rl_utils.LearningRateDecayScheduler import LearningRateDecay
from rl_utils.SAR_RNN_NReturn_RankPriority_MemoryBuffer import SAR_NStepReturn_RankPriority_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

stack_size = 4
burn_in_length = 28
trajectory_length = 100

actor_recurrent_layer_size = 256
critic_recurrent_layer_size = 256

env = gym.make('LunarLanderContinuous-v2')
state_space_shape = (stack_size, env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 2 #64
num_episodes = 5000
# actor_learning_rate = 3e-4
# critic_learning_rate = 3e-4
learning_rate = 1e-4
alpha_learning_rate = 3e-4
q_rescaling_epsilone = tf.constant(1e-6, dtype=tf.float32)
gamma = 0.99
tau = 0.005
gradient_step = 4
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(env.action_space.shape)

initializer_bounds = 3e-3

RND_SEED = 0x12345

checkpoint_step = 500
max_epoch_steps = 1000
global_step = 0

lr_scheduler = LearningRateDecay(learning_rate)
actor_optimizer = tf.keras.optimizers.Adam(lr_scheduler)
critic_optimizer = tf.keras.optimizers.Adam(lr_scheduler)

alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)
trajectory_n = tf.constant(0.9, dtype=tf.float32)

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer = SAR_NStepReturn_RankPriority_MemoryBuffer(distributed_mode=False, buffer_size=1001, N=4, gamma=gamma, 
                                                        state_shape=(stack_size, env.observation_space.shape[0]),
                                                        action_shape=env.action_space.shape, 
                                                        hidden_state_shape=(actor_recurrent_layer_size,), 
                                                        trajectory_size=trajectory_length, burn_in_length=burn_in_length,
                                                        alpha=0.7, beta=0.5, beta_increase_rate=1)

def policy_network():
    input = keras.layers.Input(shape=(state_space_shape))
    hidden_input = keras.layers.Input(shape=(actor_recurrent_layer_size))

    rnn_out, hx = keras.layers.GRU(actor_recurrent_layer_size, return_state=True)(input, initial_state=[hidden_input])
    x = keras.layers.Dense(128, activation='relu')(rnn_out)
    x = keras.layers.Dense(64, activation='relu')(x)
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, hidden_input], outputs=[mean_output, log_std_dev_output, hx])
    return model

def critic_network():
    input = keras.layers.Input(shape=(state_space_shape))
    actions_input = keras.layers.Input(shape=(outputs_count))

    # x = keras.layers.GRU(actor_recurrent_layer_size)(input)
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(actor_recurrent_layer_size)(x)
    x = keras.layers.Concatenate()([x, actions_input])

    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

@tf.function(experimental_relax_shapes=True)
def get_actions(mu, log_sigma):
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * gaus_distr.sample())

@tf.function(experimental_relax_shapes=True)
def get_log_probs(mu, sigma, actions):
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    log_probs = action_distributions.log_prob(mu + sigma * gaus_distr.sample()) - \
                tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + action_bounds_epsilon), axis=1)
    return log_probs

@tf.function(experimental_relax_shapes=True)
def train_critics(actor_hx, states, actions, next_states, rewards, gamma_powers, is_weights, dones):
    mu, log_sigma, ___ = actor([next_states, actor_hx], training=False)
    mu = tf.squeeze(mu)
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

    target_actions = get_actions(mu, log_sigma)
    target_actions_shape = tf.shape(target_actions)
    if len(target_actions_shape)  < 2:
        target_actions = tf.expand_dims(target_actions, axis=0)

    min_q = tf.math.minimum(target_critic_1([next_states, target_actions], training=False), \
                            target_critic_2([next_states, target_actions], training=False))
    min_q = tf.squeeze(min_q, axis=1)

    sigma = tf.math.exp(log_sigma)
    log_probs = get_log_probs(mu, sigma, target_actions)
    next_values = min_q - tf.math.exp(alpha_log) * log_probs

    target_q = rewards + tf.pow(gamma, gamma_powers) * (1 - dones) * next_values

    with tf.GradientTape() as tape:
        current_q = critic_1([states, actions], training=True)
        # c1_loss = mse_loss(current_q, target_q)
        c1_loss = 0.5 * tf.reduce_mean(is_weights * tf.pow(target_q - current_q, 2))
    gradients = tape.gradient(c1_loss, critic_1.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_1.trainable_variables))

    with tf.GradientTape() as tape:
        current_q = critic_2([states, actions], training=True)
        # c2_loss = mse_loss(current_q, target_q)
        c2_loss = 0.5 * tf.reduce_mean(is_weights * tf.pow(target_q - current_q, 2))
    gradients = tape.gradient(c2_loss, critic_2.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_2.trainable_variables))
    return c1_loss, c2_loss

@tf.function(experimental_relax_shapes=True)
def train_actor(states, hidden_rnn_states):
    alpha = tf.math.exp(alpha_log)
    with tf.GradientTape() as tape:
        mu, log_sigma, ___ = actor([states, hidden_rnn_states], training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma)
        target_actions_shape = tf.shape(target_actions)
        if len(target_actions_shape)  < 2:
            target_actions = tf.expand_dims(target_actions, axis=0)
        
        target_q = tf.math.minimum(critic_1([states, target_actions], training=False), \
                                   critic_2([states, target_actions], training=False))
        target_q = tf.squeeze(target_q, axis=1)
        
        sigma = tf.math.exp(log_sigma)
        log_probs = get_log_probs(mu, sigma, target_actions)

        actor_loss = tf.reduce_mean(alpha * log_probs - target_q)

        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(alpha_log * tf.stop_gradient(log_probs + target_entropy))
        alpha_gradients = alpha_tape.gradient(alpha_loss, alpha_log)
        alpha_optimizer.apply_gradients([(alpha_gradients, alpha_log)])

    gradients = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return actor_loss

@tf.function(experimental_relax_shapes=True)
def invertible_function_rescaling(self, x):
    return tf.sign(x)*(tf.sqrt(tf.abs(x) + 1) - 1) + q_rescaling_epsilone * x

@tf.function(experimental_relax_shapes=True)
def get_trajectory_error(states, actions, next_states, rewards, gamma_powers, dones, hidden_rnn_states):
    mu, log_sigma, ___ = actor([next_states, hidden_rnn_states], training=False)
    mu = tf.squeeze(mu)
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

    next_actions = get_actions(mu, log_sigma)
    next_actions_shape = tf.shape(next_actions)
    if len(next_actions_shape)  < 2:
        next_actions = tf.expand_dims(next_actions, axis=0)
    
    # Originally it was target_critic
    target_q = tf.math.minimum(critic_1([next_states, next_actions], training=False), \
                               critic_2([next_states, next_actions], training=False))

    # Vanila td_error calculation way.
    # sigma = tf.math.exp(log_sigma)
    # log_probs = get_log_probs(mu, sigma, next_actions)
    # next_values = target_q - tf.math.exp(alpha_log) * log_probs
    # target_q = rewards + tf.pow(gamma, gamma_powers) * (1 - dones) * next_values

    # Simplified td_error calculation way. Tend to stuck in local minima
    inverse_q_rescaling = tf.math.pow(invertible_function_rescaling(tf.squeeze(target_q, axis=1)), -1)
    target_q = rewards + tf.math.pow(gamma, gamma_powers + 1) * (1 - dones) * inverse_q_rescaling
    target_q = invertible_function_rescaling(target_q)

    current_q = tf.math.minimum(critic_1([states, actions], training=False), \
                                critic_2([states, actions], training=False))

    td_errors = target_q - tf.squeeze(current_q, axis=1)
    td_errors_shape = tf.shape(td_errors)
    if len(td_errors_shape) == 0:
        return td_errors

    return tf.reduce_max(td_errors) * trajectory_n + (1-trajectory_n)*tf.reduce_mean(td_errors)

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

@tf.function(experimental_relax_shapes=True)
def actor_burn_in(states, hx0, trajectory_length):
    hx = hx0 # hx = tf.expand_dims(hx0, axis=0)
    for s in states:
        _, __, hx = actor([tf.expand_dims(s, axis = 0), hx], training=False)
    return tf.tile(hx, [trajectory_length, 1])

def __interpolation_step(s0, action, stack_size=4):
    result_states = []
    sN, r, d, _ = env.step(action)
    #interpolate between s0 and sN
    xp = [0, stack_size - 1]
    x = [i for i in range(stack_size) if i not in xp]
    interp_count = stack_size - 2
    result_states.append(s0)
    for _ in range(interp_count):
        result_states.append(np.zeros(shape=(len(s0)),dtype=np.float))
    result_states.append(sN)
    for i , y_boundary in enumerate(zip(s0, sN)):
        y_linear = np.interp(x, xp, y_boundary)
        for j, y in enumerate(y_linear):
            result_states[j+1][i] = y
    return result_states, r, d

actor = policy_network()

critic_1 = critic_network()
target_critic_1 = critic_network()
target_critic_1.set_weights(critic_1.get_weights())

critic_2 = critic_network()
target_critic_2 = critic_network()
target_critic_2.set_weights(critic_2.get_weights())

rewards_history = []

for i in range(num_episodes):
    done = False
    state0 = env.reset()
    observation = []
    for _ in range(stack_size):
        observation.append(state0)

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []

    actor_hx = tf.zeros(shape=(1, actor_recurrent_layer_size), dtype=tf.float32)
    start_time = datetime.now()

    while not done:
        #env.render()
        mean, log_std_dev, actor_hx = actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])

        next_observation, reward, done = __interpolation_step(state0, [throttle_action, eng_ctrl_action])
        state0 = next_observation[-1:][0]

        exp_buffer.store(actor_hx, observation, [throttle_action, eng_ctrl_action], reward, float(done))

        if len(exp_buffer) > burn_in_length:
            td_errors = dict()
            meta_idxs = list()
            # get one trajectory at a time
            for actor_h, burn_in_states, states, actions, next_states, rewards, gamma_powers, dones, is_weights, meta_idx in exp_buffer.sample(batch_size):
                actor_training_hx = actor_burn_in(burn_in_states, actor_h, tf.convert_to_tensor(len(rewards), dtype=tf.int32))
                meta_idxs.append(meta_idx)
                for _ in range(gradient_step):
                    critic1_loss, critic2_loss = train_critics(actor_training_hx, states, actions, next_states, rewards, gamma_powers, is_weights, dones)
                    critic_loss_history.append(critic1_loss)
                    critic_loss_history.append(critic2_loss)
                
                    actor_loss = train_actor(states, actor_training_hx)
                    actor_loss_history.append(actor_loss)
                td_errors[meta_idx] = get_trajectory_error(states, actions, next_states, rewards, gamma_powers, dones, actor_training_hx)
                soft_update_models()
            meta_idxs.sort(reverse=True) # ensure reversed order of updaing errors
            exp_buffer.update_priorities(meta_idxs, [td_errors[idx] for idx in meta_idxs])

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    if episodic_reward > 50 and epoch_steps < 900:
        lr_scheduler.decay()

    delta_time = datetime.now() - start_time
    episode_minutes = int(delta_time.seconds / 60)
    episode_seconds = delta_time.seconds - episode_minutes * 60
    print(f'[epoch {i} ({epoch_steps}) {episode_minutes}:{episode_seconds}] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
input("training complete...")
