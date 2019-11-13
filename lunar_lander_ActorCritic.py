import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 5000
learning_rate = 0.001
X_shape = (env.observation_space.shape[0])
gamma = 0.97

checkpoint_step = 500

outputs_count = env.action_space.n

checkpoint_file_name = 'll_ac_checkpoint.h5'

optimizer = tf.keras.optimizers.Adam(learning_rate)

def policy_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    actions = keras.layers.Dense(outputs_count, activation='softmax')(x)
    v= keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=[actions,v])
    return model

@tf.function
def learn(state, action, reward, new_state, done):
    one_hot_actions_mask = tf.one_hot(action, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) # shape = len(actions), 4

    target = reward
    if done==0:
        _, v_next = policy(tf.expand_dims(new_state, axis=0))
        target += gamma * v_next

    with tf.GradientTape() as tape:
        actions_distribution, v = policy(tf.expand_dims(state, axis = 0))
        delta = target - v

        actor_loss = tf.reduce_sum(-tf.math.log(actions_distribution) * one_hot_actions_mask) * delta
        crtitic_loss = tf.math.square(delta)

        loss = actor_loss + crtitic_loss
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

    while not done:
        #env.render()
        actions_distribution, _ = policy(np.expand_dims(observation, axis = 0)) #actions distribution frequently is all NaN!!!
        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution[0].numpy())
        next_observation, reward, done, _ = env.step(chosen_action)

        state_tensor = tf.convert_to_tensor(observation, dtype = tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_observation, dtype = tf.float32)
        action_tensor = tf.convert_to_tensor(chosen_action, dtype = tf.uint8)
        reward_tensor = tf.convert_to_tensor(reward, dtype = tf.float32)
        done_tensor = tf.convert_to_tensor(int(done), dtype = tf.uint8)
        loss = learn(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)[0][0].numpy()

        episod_rewards.append(reward)
        epoch_steps+=1
        observation = next_observation

    if i % checkpoint_step == 0 and i > 0:
        policy.save(checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} (steps: {epoch_steps})] Loss: {loss:.4f} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
policy.save('lunar_lander_ac.h5')
input("training complete...")

