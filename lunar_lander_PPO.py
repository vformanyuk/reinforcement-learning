import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 50000
actor_learning_rate = 0.0001
critic_learning_rate = 0.0005
clipping_epsilon = 0.2
batch_size = 2048
X_shape = (env.observation_space.shape[0])
gamma = 0.99
gae_lambda = 0.95
entropy_beta = 0.001

lambda_gamma_constant = tf.constant(gae_lambda * gamma, dtype=tf.float32)

checkpoint_step = 500
max_epoch_steps = 300 #lunar lender tends to stuck with 1000 steps episode length
train_epoches = 4

outputs_count = env.action_space.n

actor_checkpoint_file_name = 'll_ppo_actor_checkpoint.h5'
critic_checkpoint_file_name = 'll_ppo_critic_checkpoint.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

tf.random.set_seed(0x12345)
np.random.random(0)

def policy_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def value_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(512, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x) #, kernel_regularizer=keras.regularizers.l2(0.001))(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

@tf.function
def train_actor(states, actions, target_distributions, adv):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32)

    with tf.GradientTape() as tape:
        action_logits = tf.squeeze(evaluation_policy(states, training=True))
        evalution_distribution = tf.nn.softmax(action_logits)

        #with tape.stop_recording():
        #    evalution_log_distribution = tf.nn.log_softmax(action_logits)
        #    entropy = -tf.reduce_sum(evalution_log_distribution * evalution_distribution)

        r = tf.reduce_sum(evalution_distribution * one_hot_actions_mask, axis=1) / target_distributions
        r_clipped = tf.clip_by_value(r, 1 - clipping_epsilon, 1 + clipping_epsilon)
        loss = -tf.reduce_mean(tf.math.minimum(r * adv, r_clipped * adv)) #+ entropy_beta * entropy
    gradients = tape.gradient(loss, evaluation_policy.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, evaluation_policy.trainable_variables))
    return loss

# tf.function can not define variables
v_target = tf.Variable(0., dtype = tf.float32, trainable=False)
gae = tf.Variable(0., dtype = tf.float32, trainable=False)

@tf.function
def train_critic(states, rewards, dones):
    V_target_tensor = tf.TensorArray(dtype = tf.float32, size = batch_size)
    gae_tensor = tf.TensorArray(dtype = tf.float32, size = batch_size)
    
    tensor_idx = batch_size - 1
    gae.assign(0.)
    v_target.assign(0.)
    
    end_idx = len(rewards) - 1

    with tf.GradientTape() as tape:
        V = critic(states, training=True)

        for j in tf.range(end_idx, -1, delta = -1):
            V_next = V[j+1] if (j+1) <= end_idx else tf.constant(0., dtype=tf.float32, shape=(1,))
            delta = rewards[j] + gamma * V_next * (1-dones[j]) - V[j]
            
            current_gae = gae.assign(tf.squeeze(delta) + lambda_gamma_constant * gae.value())
            current_v_target = v_target.assign(rewards[j] + gamma * v_target.value() * (1-dones[j]))
            gae_tensor = gae_tensor.write(tensor_idx, current_gae)
            V_target_tensor = V_target_tensor.write(tensor_idx, current_v_target)
            
            tensor_idx -= 1 #filling tensor array from behind, so no need to reverse

        advantage = gae_tensor.stack()
        #advantage = (advantage - tf.reduce_mean(advantage)) / tf.math.reduce_std(advantage)

        v_target_ = V_target_tensor.stack()
        #v_target_ = (v_target_ - tf.reduce_mean(v_target_)) / tf.math.reduce_std(v_target_)

        loss = 0.5 * mse_loss(v_target_, V)
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, advantage

if os.path.isfile(actor_checkpoint_file_name):
    target_policy = keras.models.load_model(checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    target_policy = policy_network()
    print("New model created.")

if os.path.isfile(critic_checkpoint_file_name):
    critic = keras.models.load_model(critic_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic = value_network()
    print("New Critic model created.")

evaluation_policy = policy_network()
evaluation_policy.set_weights(target_policy.get_weights())

states_memory = []
rewards_memory = []
actions_memory = []
action_prob_memory = []
terminal_memory = []

rewards_history = []
global_step = 0

for i in range(num_episodes):
    done = False
    observation = env.reset()
    episod_rewards = []
    epoch_steps = 0

    while not done and epoch_steps <= max_epoch_steps:
        #env.render()
        actions_logits = target_policy(np.expand_dims(observation, axis = 0), training=False)
        actions_logits = tf.squeeze(actions_logits)
        actions_distribution = tf.nn.softmax(actions_logits).numpy()

        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        next_observation, reward, done, _ = env.step(chosen_action)

        episod_rewards.append(reward)
        rewards_memory.append(reward)
        states_memory.append(tf.convert_to_tensor(observation, dtype=tf.float32))
        actions_memory.append(chosen_action)
        action_prob_memory.append(actions_distribution[chosen_action])
        terminal_memory.append(done)

        epoch_steps += 1
        global_step += 1

        # obtain trajectory segment and train networks
        if global_step >= batch_size:
            critic_loss_history = []
            actor_loss_history = []
            for _ in range(train_epoches):
                critic_loss, adv = train_critic(tf.stack(states_memory),
                                                tf.convert_to_tensor(rewards_memory, dtype=tf.float32), 
                                                tf.convert_to_tensor(terminal_memory, dtype=tf.float32))
                critic_loss_history.append(critic_loss)
                actor_loss = train_actor(tf.stack(states_memory),
                                         tf.convert_to_tensor(actions_memory, dtype=tf.int32), 
                                         tf.convert_to_tensor(action_prob_memory, dtype=tf.float32),
                                         adv)
                actor_loss_history.append(actor_loss)

            states_memory.clear()
            rewards_memory.clear()
            terminal_memory.clear()
            actions_memory.clear()
            action_prob_memory.clear()

            target_policy.set_weights(evaluation_policy.get_weights())
            global_step = 0
            print(f'=============  Actor loss: {np.mean(actor_loss_history):.4f} Critic loss: {np.mean(critic_loss_history):.4f}  =============')
        observation = next_observation

    #if i % checkpoint_step == 0 and i > 0:
    #    policy.save(checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
target_policy.save('lunar_ppo.h5')
input("training complete...")
