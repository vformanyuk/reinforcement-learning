import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 5000
learning_rate = 0.001
X_shape = (env.observation_space.shape[0])
discount_factor = 0.97

checkpoint_step = 500

outputs_count = env.action_space.n

checkpoint_file_name = 'll_pgrad_checkpoint.h5'

optimizer = tf.keras.optimizers.Adam(learning_rate)

def policy_network():
    input = keras.layers.Input(shape=(None,X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(outputs_count, activation='softmax')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

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

@tf.function(experimental_relax_shapes=True) # because each episode lenght varies Tensorflow will retrace this function whenever input arguments shapes changes
def learn(states, actions, G):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) # shape = len(actions), 4

    with tf.GradientTape() as tape:
        action_distributions = policy(states)
        neg_log_like = tf.reduce_sum(-tf.math.log(action_distributions) * one_hot_actions_mask, axis=1) # shape (len(actions),4) => (len(actions),1)

        loss = tf.reduce_sum(tf.math.multiply(neg_log_like,G)) # sum(-logPi(a|s)*G), G shape = (len(actions),1)
    gradients = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy.trainable_variables))
    return loss

if os.path.isfile(checkpoint_file_name):
    policy = keras.models.load_model(checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    policy = policy_network()
    print("New model created.")

np.random.random(0)
rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()
    epoch_steps = 0
    episod_rewards = []
    actions_memory = []
    states_memory = []
    nans = 0

    while not done:
        #env.render()
        actions_distribution = policy(np.expand_dims(observation, axis = 0))[0].numpy()
        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        next_observation, reward, done, _ = env.step(chosen_action)

        episod_rewards.append(reward)
        actions_memory.append(chosen_action)
        states_memory.append(tf.convert_to_tensor(observation, dtype=tf.float32))
        epoch_steps+=1
        observation = next_observation

    G = tf.convert_to_tensor(discounted_rewards(episod_rewards), dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(actions_memory, dtype=tf.uint8)
    states_tensor = tf.stack(states_memory)
    loss = learn(states_tensor, actions_tensor, G)

    if i % checkpoint_step == 0 and i > 0:
        policy.save(checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} (steps: {epoch_steps})] Loss: {loss} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
policy.save('lunar_policy_grad.h5')
input("training complete...")
