import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from rl_utils import SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')
X_shape = (env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 100
num_episodes = 5000
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
gamma = 0.99
tau = 0.0005
delay_step = 2

RND_SEED = 0x12345

checkpoint_step = 500
max_epoch_steps = 1000
global_step = 0

actor_checkpoint_file_name = 'll_td3_actor_checkpoint.h5'
critic_1_checkpoint_file_name = 'll_td3_critic1_checkpoint.h5'
critic_2_checkpoint_file_name = 'll_td3_critic2_checkpoint.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(x)
    #x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(outputs_count, activation='tanh',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def critic_network():
    input = keras.layers.Input(shape=(X_shape))
    actions_input = keras.layers.Input(shape=(outputs_count))

    x = keras.layers.Concatenate()([input, actions_input])

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.BatchNormalization()(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                kernel_regularizer = keras.regularizers.l2(0.01),
                                bias_regularizer = keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

@tf.function
def train_critics(states, actions, next_states, rewards, dones):
    noise = tf.random.normal(shape=(batch_size, outputs_count), mean=0.0, stddev = 0.2)
    clipped_noise = tf.clip_by_value(noise, -0.5, 0.5)
    target_mu = target_policy(next_states, training=False) + clipped_noise

    min_critic = tf.math.minimum(target_critic_1([next_states, target_mu], training=False), 
                                 target_critic_2([next_states, target_mu], training=False))
    target_q = rewards + gamma * tf.reduce_max((1 - dones) * min_critic, axis = 1)

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
    with tf.GradientTape() as tape:
        current_mu = actor(states, training=True)
        current_q = critic_1([states, current_mu], training=False)
        a_loss = tf.reduce_mean(-current_q)
    gradients = tape.gradient(a_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return a_loss


def soft_update_models():
    target_actor_weights = target_policy.get_weights()
    actor_weights = actor.get_weights()
    updated_actor_weights = []
    for aw,taw in zip(actor_weights, target_actor_weights):
        updated_actor_weights.append(tau * aw + (1.0 - tau) * taw)
    target_policy.set_weights(updated_actor_weights)

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
    actor = policy_network()
    print("New model created.")
target_policy = policy_network()
target_policy.set_weights(actor.get_weights())

if os.path.isfile(critic_1_checkpoint_file_name):
    critic_1 = keras.models.load_model(critic_1_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic_1 = critic_network()
    print("New Critic model created.")
target_critic_1 = critic_network()
target_critic_1.set_weights(critic_1.get_weights())

if os.path.isfile(critic_2_checkpoint_file_name):
    critic_2 = keras.models.load_model(critic_2_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic_2 = critic_network()
    print("New Critic model created.")
target_critic_2 = critic_network()
target_critic_2.set_weights(critic_2.get_weights())

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
        chosen_action = actor(np.expand_dims(observation, axis = 0), training=False)[0].numpy() + np.random.normal(0,0.1)#np.clip(actions_noise, -0.5, 0.5)
        next_observation, reward, done, _ = env.step(chosen_action)

        exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

        if global_step > 10 * batch_size:
            states, actions, next_states, rewards, dones = exp_buffer(batch_size)
            critic1_loss, critic2_loss = train_critics(states, actions, next_states, rewards, dones)
            critic_loss_history.append(critic1_loss)
            critic_loss_history.append(critic2_loss)
            
            if global_step % delay_step == 0:
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
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_td3.h5')
env.close()
input("training complete...")
