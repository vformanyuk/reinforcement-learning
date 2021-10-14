import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
import tensorflow_probability as tfp
from tensorflow import keras
from rl_utils.SAR_RNN_MemoryBuffer import SAR_NStepReturn_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

stack_size = 4
burn_in_length = 10

actor_recurrent_layer_size = 256
critic_recurrent_layer_size = 256

env = gym.make('LunarLanderContinuous-v2')
state_space_shape = (stack_size, env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 128
num_episodes = 5000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
gamma = 0.99
tau = 0.005
gradient_step = 3
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
alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = SAR_NStepReturn_RandomAccess_MemoryBuffer(exp_buffer_capacity, 4, 0.99, state_space_shape, env.action_space.shape, 
                                                        trajectory_size=40, trajectory_overlap=20, burn_in_length=burn_in_length)

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
def train_critics(actor_hx, states, actions, next_states, rewards, gamma_powers, dones):
    mu, log_sigma, ___ = actor([next_states, actor_hx], training=False)
    mu = tf.squeeze(mu)
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

    target_actions = get_actions(mu, log_sigma)

    min_q = tf.math.minimum(target_critic_1([next_states, target_actions], training=False), \
                            target_critic_2([next_states, target_actions], training=False))
    min_q = tf.squeeze(min_q, axis=1)

    sigma = tf.math.exp(log_sigma)
    log_probs = get_log_probs(mu, sigma, target_actions)
    next_values = min_q - tf.math.exp(alpha_log) * log_probs

    target_q = rewards + tf.pow(gamma, gamma_powers) * (1 - dones) * next_values

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

@tf.function(experimental_relax_shapes=True)
def train_actor(states, hidden_rnn_states):
    alpha = tf.math.exp(alpha_log)
    with tf.GradientTape() as tape:
        mu, log_sigma, ___ = actor([states, hidden_rnn_states], training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma)
        
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

def actor_burn_in(states, hx0, trajectory_length):
    hx = hx0
    for s in states:
        _, __, hx = actor([np.expand_dims(s, axis = 0), hx], training=False)
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

    while not done:
        #env.render()
        mean, log_std_dev, actor_hx = actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])

        next_observation, reward, done = __interpolation_step(state0, [throttle_action, eng_ctrl_action])
        state0 = next_observation[-1:][0]

        exp_buffer.store(actor_hx, observation, [throttle_action, eng_ctrl_action], reward, float(done))

        if global_step > 4 * batch_size:
            # get one trajectory at a time
            for actor_h, burn_in_states, states, actions, next_states, rewards, gamma_powers, dones in exp_buffer(batch_size):
                actor_training_hx = actor_burn_in(burn_in_states, actor_h, len(rewards))
                for _ in range(gradient_step):
                    critic1_loss, critic2_loss = train_critics(actor_training_hx, states, actions, next_states, rewards, gamma_powers, dones)
                    critic_loss_history.append(critic1_loss)
                    critic_loss_history.append(critic2_loss)
                
                    actor_loss = train_actor(states, actor_training_hx)
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
env.close()
input("training complete...")
