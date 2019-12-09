import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import deque

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 5000
learning_rate = 0.001
clipping_epsilon = 0.2
batch_size = 64
X_shape = (env.observation_space.shape[0])
discount_factor = 0.97

checkpoint_step = 500
copy_step = 50
steps_train = 4
start_steps = 1000
global_step = 0

outputs_count = env.action_space.n

checkpoint_file_name = 'll_ppo_checkpoint.h5'

optimizer = tf.keras.optimizers.Adam(learning_rate)

exp_buffer = deque(maxlen=200000)

def policy_network():
    input = keras.layers.Input(shape=(None,X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(outputs_count, activation='softmax')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

@tf.function
def learn(states, actions, target_probs, adv):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) # shape = len(actions), 4

    with tf.GradientTape() as tape:
        action_distributions = evaluation_policy(states, training=True)
        evaluation_probs = tf.reduce_sum(action_distributions * one_hot_actions_mask, axis=1)
        r = evaluation_probs / target_probs
        r_clipped = tf.clip_by_value(r, 1 - clipping_epsilon, 1 + clipping_epsilon)
        loss = tf.reduce_mean(tf.math.multiply(tf.math.minimum(r, r_clipped), adv))
    gradients = tape.gradient(loss, evaluation_policy.trainable_variables)
    optimizer.apply_gradients(zip(gradients, evaluation_policy.trainable_variables))
    return loss

def sample_expirience(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    return np.array(exp_buffer)[perm_batch]

def discounted_rewards(episode_rewards):
    G = np.zeros_like(episode_rewards, dtype=np.float32)
    for i in range(len(episode_rewards)):
        G_sum = 0
        discount = 1
        for j in range(i, len(episode_rewards)):
            G_sum += episode_rewards[j] * discount
            discount *= discount_factor
        G[i] = G_sum
    mean = np.mean(G)
    std_dev = np.std(G)
    G = (G - mean) / (std_dev if std_dev > 0 else 1)
    return G

if os.path.isfile(checkpoint_file_name):
    target_policy = keras.models.load_model(checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    target_policy = policy_network()
    print("New model created.")

evaluation_policy = policy_network()
evaluation_policy.set_weights(target_policy.get_weights())

tf.random.set_seed(0x12345)
np.random.random(0)
rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()
    epoch_steps = 0

    while not done:
        #env.render()
        actions_distribution = target_policy(np.expand_dims(observation, axis = 0), training=False)[0].numpy()
        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        next_observation, reward, done, _ = env.step(chosen_action)

        exp_buffer.append([tf.convert_to_tensor(observation, dtype=tf.float32), chosen_action, actions_distribution[chosen_action]])

        if global_step % steps_train == 0 and global_step > start_steps:
            samples = sample_expirience(batch_size)

        if (global_step + 1) % copy_step == 0 and global_step > start_steps:
            target_policy.set_weights(evaluation_policy.get_weights())

        epoch_steps += 1
        global_step += 1
        observation = next_observation

    #G = tf.convert_to_tensor(discounted_rewards(episod_rewards), dtype=tf.float32)
    #actions_tensor = tf.convert_to_tensor(actions_memory, dtype=tf.uint8)
    #states_tensor = tf.stack(states_memory)
    #loss = learn(states_tensor, actions_tensor, G)

    #if i % checkpoint_step == 0 and i > 0:
    #    policy.save(checkpoint_file_name)

    #total_episod_reward = sum(episod_rewards)
    #rewards_history.append(total_episod_reward)

    #last_mean = np.mean(rewards_history[-100:])
    #print(f'[epoch {i} (steps: {epoch_steps})] Loss: {loss} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    #if last_mean > 200:
    #    break
env.close()
policy.save('lunar_policy_grad.h5')
input("training complete...")
