import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from rl_utils import OUActionNoise, SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')
X_shape = (env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 64
num_episodes = 5000
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
gamma = 0.99
tau = 0.001

RND_SEED = 0x12345

checkpoint_step = 500
max_epoch_steps = 1000
global_step = 0
steps_train = 4

actor_checkpoint_file_name = 'll_ddpg_actor_checkpoint.h5'
critic_checkpoint_file_name = 'll_ddpg_critic_checkpoint.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

action_noise = OUActionNoise(mu=np.zeros(outputs_count))

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
    actions_input = keras.layers.Input(shape=(outputs_count))
    input = keras.layers.Input(shape=(X_shape))

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Concatenate()([x, actions_input])
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
def train_actor_critic(states, actions, next_states, rewards, dones):
    target_mu = target_policy(next_states, training=False)
    target_q = rewards + gamma * tf.reduce_max((1 - dones) * target_critic([next_states, target_mu], training=False), axis = 1)

    with tf.GradientTape() as tape:
        current_q = critic([states, actions], training=True)
        c_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    with tf.GradientTape() as tape:
        current_mu = actor(states, training=True)
        current_q = critic([states, current_mu], training=False)
        a_loss = tf.reduce_mean(-current_q)
    gradients = tape.gradient(a_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return a_loss, c_loss

def soft_update_models():
    target_actor_weights = target_policy.get_weights()
    actor_weights = actor.get_weights()
    updated_actor_weights = []
    for aw,taw in zip(actor_weights,target_actor_weights):
        updated_actor_weights.append(tau * aw + (1.0 - tau) * taw)
    target_policy.set_weights(updated_actor_weights)

    target_critic_weights = target_critic.get_weights()
    critic_weights = critic.get_weights()
    updated_critic_weights = []
    for cw,tcw in zip(critic_weights,target_critic_weights):
        updated_critic_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic.set_weights(updated_critic_weights)

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    actor = policy_network()
    print("New model created.")

if os.path.isfile(critic_checkpoint_file_name):
    critic = keras.models.load_model(critic_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic = critic_network()
    print("New Critic model created.")

target_policy = policy_network()
target_policy.set_weights(actor.get_weights())

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
        chosen_action = actor(np.expand_dims(observation, axis = 0), training=False)[0].numpy() + action_noise()
        next_observation, reward, done, _ = env.step(chosen_action)

        exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

        if global_step > 10 * batch_size and global_step % steps_train == 0:
            actor_loss, critic_loss = train_actor_critic(*exp_buffer(batch_size))

            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            
            soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    if i % checkpoint_step == 0 and i > 0:
        actor.save(actor_checkpoint_file_name)
        critic.save(critic_checkpoint_file_name)

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_ddpg.h5')
env.close()
input("training complete...")