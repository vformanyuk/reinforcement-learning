import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import os

from env.lunar_lander import LunarLander
from rl_utils.ad_layer import ADLayer, kWTA_Layer
from rl_utils.SARST_RandomAccess_MemoryBuffer import SARST_MultiTask_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = LunarLander(continuous=True) #gym.make('LunarLanderContinuous-v2')
X_shape = (env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]
context_vector_length = 2
dendrits_count = 2

batch_size = 100
num_episodes = 5000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
gamma = 0.99
tau = 0.005
gradient_step = 1
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(env.action_space.shape)

initializer_bounds = 3e-3

RND_SEED = 0x12345

checkpoint_step = 5
max_epoch_steps = 1000
global_step = 0

actor_checkpoint_file_name = 'll_sac_actor_checkpoint_mt.h5'
critic_1_checkpoint_file_name = 'll_sac_critic1_checkpoint_mt.h5'
critic_2_checkpoint_file_name = 'll_sac_critic2_checkpoint_mt.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)

land_task = tf.constant([1,0], dtype=tf.float32)
lift_off_task = tf.constant([0,1], dtype=tf.float32)

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = SARST_MultiTask_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape, context_vector_length)

def policy_network(debug=False):
    input = keras.layers.Input(shape=(X_shape))
    context_intput = keras.layers.Input(shape=(context_vector_length))

    x = keras.layers.Dense(512, activation='linear', name="actr_dense_1")(input)
    x = kWTA_Layer(top_activations_count=128, name="actr_kwta_1")(x) # takes top 25% of neurons, non-linearity layer
    x = ADLayer(256, dendrits_count, context_vector_length, use_abs_max = False, name="actr_ad_1")([x, context_intput])
    x = kWTA_Layer(top_activations_count=64, name="actr_kwta_2")(x) # takes top 25% of neurons, non-linearity layer
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, context_intput], outputs=[mean_output, log_std_dev_output])
    model.run_eagerly = debug
    return model

def critic_network(debug=False):
    input = keras.layers.Input(shape=(X_shape))
    context_intput = keras.layers.Input(shape=(context_vector_length))
    actions_input = keras.layers.Input(shape=(outputs_count))

    x = keras.layers.Concatenate()([input, actions_input])
    x = keras.layers.Dense(512, activation='linear', name="crtc_dense_1")(x)
    x = kWTA_Layer(top_activations_count=128, name="crtc_kwta_1")(x) # takes top 25% of neurons, non-linearity layer
    x = ADLayer(512, dendrits_count, context_vector_length, use_abs_max = False, name="crtc_ad_1")([x, context_intput])
    x = kWTA_Layer(top_activations_count=128, name="crtc_kwta_2")(x) # takes top 25% of neurons, non-linearity layer
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input, context_intput], outputs=q_layer)
    model.run_eagerly = debug
    return model

@tf.function
def get_actions(mu, log_sigma, noise = None):
    if noise == None:
        noise = gaus_distr.sample()
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

@tf.function
def get_log_probs(mu, sigma, actions, noise):
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    log_probs = action_distributions.log_prob(mu + sigma * noise) - \
                tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + action_bounds_epsilon), axis=1)
    return log_probs

@tf.function
def train_critics(states, actions, next_states, rewards, dones, context_vectors):
    mu, log_sigma = actor([next_states, context_vectors])
    mu = tf.squeeze(mu)
    noise = gaus_distr.sample(sample_shape=(batch_size, 2))
    log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

    target_actions = get_actions(mu, log_sigma, noise)

    min_q = tf.math.minimum(target_critic_1([next_states, target_actions, context_vectors], training=False), \
                            target_critic_2([next_states, target_actions, context_vectors], training=False))
    min_q = tf.squeeze(min_q, axis=1)

    sigma = tf.math.exp(log_sigma)
    log_probs = get_log_probs(mu, sigma, target_actions, noise)
    next_values = min_q - tf.math.exp(alpha_log) * log_probs # min(Q1^,Q2^) - alpha * logPi

    target_q = rewards + gamma * (1 - dones) * next_values

    with tf.GradientTape() as tape:
        current_q = critic_1([states, actions, context_vectors], training=True)
        c1_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c1_loss, critic_1.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_1.trainable_variables))

    with tf.GradientTape() as tape:
        current_q = critic_2([states, actions, context_vectors], training=True)
        c2_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c2_loss, critic_2.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic_2.trainable_variables))
    return c1_loss, c2_loss

@tf.function
def train_actor(states, context_vectors):
    alpha = tf.math.exp(alpha_log)
    noise = gaus_distr.sample(sample_shape=(batch_size, 2))
    with tf.GradientTape() as tape:
        mu, log_sigma = actor([states, context_vectors], training=True)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

        target_actions = get_actions(mu, log_sigma, noise)
        
        target_q = tf.math.minimum(critic_1([states, target_actions, context_vectors], training=False), \
                                   critic_2([states, target_actions, context_vectors], training=False))
        target_q = tf.squeeze(target_q, axis=1)
        
        sigma = tf.math.exp(log_sigma)
        log_probs = get_log_probs(mu, sigma, target_actions, noise)

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

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name, custom_objects={'ADLayer': ADLayer , "kWTA_Layer" : kWTA_Layer})
    print("Model restored from checkpoint.")
else:
    actor = policy_network()
    print("New model created.")

if os.path.isfile(critic_1_checkpoint_file_name):
    critic_1 = keras.models.load_model(critic_1_checkpoint_file_name, custom_objects={'ADLayer': ADLayer , 'kWTA_Layer' : kWTA_Layer})
    print("Critic model restored from checkpoint.")
else:
    critic_1 = critic_network()
    print("New Critic model created.")
target_critic_1 = critic_network()
target_critic_1.set_weights(critic_1.get_weights())

if os.path.isfile(critic_2_checkpoint_file_name):
    critic_2 = keras.models.load_model(critic_2_checkpoint_file_name, custom_objects={'ADLayer': ADLayer , 'kWTA_Layer' : kWTA_Layer})
    print("Critic model restored from checkpoint.")
else:
    critic_2 = critic_network()
    print("New Critic model created.")
target_critic_2 = critic_network()
target_critic_2.set_weights(critic_2.get_weights())

landing_rewards_history = []
lift_off_rewards_history = []
training_complete = False

for i in range(num_episodes):
    done = False
    lift_off = np.random.uniform() > 0.5
    observation = env.reset(lift_off=lift_off)

    context_vector = land_task if not lift_off else lift_off_task
    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []

    while not done:
        #env.render()
        mean, log_std_dev = actor([np.expand_dims(observation, axis = 0), np.expand_dims(context_vector, axis = 0)], training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])

        next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])

        exp_buffer.store(context_vector, observation, [throttle_action, eng_ctrl_action], next_observation, reward, float(done))

        if global_step > 10 * batch_size:
            states, actions, next_states, rewards, dones, context_vectors = exp_buffer(batch_size)

            for _ in range(gradient_step):
                critic1_loss, critic2_loss = train_critics(states, actions, next_states, rewards, dones, context_vectors)
                critic_loss_history.append(critic1_loss)
                critic_loss_history.append(critic2_loss)
            
                actor_loss = train_actor(states, context_vectors)
                actor_loss_history.append(actor_loss)
            soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    # if i % checkpoint_step == 0 and i > 0:
    #     actor.save(actor_checkpoint_file_name)
    #     critic_1.save(critic_1_checkpoint_file_name)
    #     critic_2.save(critic_2_checkpoint_file_name)

    if lift_off:
        lift_off_rewards_history.append(episodic_reward)
    else:
        landing_rewards_history.append(episodic_reward)
    lift_off_last_mean = np.mean(lift_off_rewards_history[-100:])
    landing_last_mean = np.mean(landing_rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean100_landing={landing_last_mean:.4f} Mean100_lift={lift_off_last_mean:.4f}')
    if lift_off_last_mean > 200 and lift_off_last_mean > 200:
        training_complete = True
        break
if training_complete:
    actor.save('lunar_lander_sac_multitask.h5')
env.close()
input("training complete...")
