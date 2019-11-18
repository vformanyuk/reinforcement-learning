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
gamma = 0.99

checkpoint_step = 500

outputs_count = env.action_space.n

checkpoint_file_name = 'll_ac_checkpoint.h5'

optimizer = tf.keras.optimizers.Adam(learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

def policy_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(input)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    actions_layer = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=actions_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(256, activation='relu')(input) #, kernel_regularizer=keras.regularizers.l2(0.01)
    x = keras.layers.Dense(128, activation='relu')(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

actor = policy_network()
critic = value_network()

#@tf.function(experimental_relax_shapes=True)
def learn(states, actions, rewards):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) # shape = len(actions), 4
    advantage = tf.Variable(initial_value = tf.zeros_like(rewards), dtype=tf.float32, trainable=False)

    #critic training
    with tf.GradientTape() as tape:
        Q_tensor = tf.TensorArray(dtype = tf.float32, size = len(rewards))
        values = critic(states, training=True)
        values = tf.squeeze(values)
        
        # 1. Calculate episode Q value
        with tape.stop_recording():
            Q_target = 0.
            rewards_max_idx = len(rewards) - 1
            for j in tf.range(rewards_max_idx, 0, delta = -1):
                Q_target = rewards[j] + gamma*Q_target
                Q_tensor.write(rewards_max_idx - j,  Q_target)

        # 2. Calculate advantage. Q(s,a) - V(s)
        advantage.assign(Q_tensor.stack() - values)

        crtitic_loss = mse_loss(Q_tensor.stack(), values)
    critic_gradients = tape.gradient(crtitic_loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

    # actor training
    with tf.GradientTape() as tape:
        actions_logits = actor(states, training=True)
        actions_distribution = tf.nn.log_softmax(actions_logits)
        
        actor_loss = tf.reduce_mean(-tf.reduce_sum(actions_distribution * one_hot_actions_mask, axis=1) * advantage)
    actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
    return actor_loss + crtitic_loss

#if os.path.isfile(checkpoint_file_name):
#    policy = keras.models.load_model(checkpoint_file_name)
#    print("Model restored from checkpoint.")
#else:
#    policy = policy_network()
#    print("New model created.")

np.random.random(0)
rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()
    epoch_steps = 0
    episod_rewards = []
    states_memory = []
    actions_memory = []

    while not done:
        actions_logits = actor(np.expand_dims(observation, axis = 0), training=False)
        actions_distribution = tf.nn.softmax(actions_logits)[0].numpy()

        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        next_observation, reward, done, _ = env.step(chosen_action)

        episod_rewards.append(reward)
        actions_memory.append(chosen_action)
        states_memory.append(tf.convert_to_tensor(observation, dtype = tf.float32))

        epoch_steps+=1
        observation = next_observation

    states_tensor = tf.stack(states_memory)
    actions_tensor = tf.convert_to_tensor(actions_memory, dtype = tf.int32)
    rewards_tensor = tf.convert_to_tensor(episod_rewards, dtype = tf.float32)
    loss = learn(states_tensor, actions_tensor, rewards_tensor).numpy()

    #if i % checkpoint_step == 0 and i > 0:
    #    policy.save(checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} (steps: {epoch_steps})] Loss: {loss:.4f} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
actor.save('lunar_lander_ac.h5')
input("training complete...")

