import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 800
learning_rate = 0.001
batch_size = 48
X_shape = (env.observation_space.shape[0])
discount_factor = 0.99

global_step = 0
copy_step = 50
steps_train = 4
start_steps = 1000

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 1.5e-4

outputs_count = env.action_space.n

optimizer = tf.keras.optimizers.Adam(learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

def q_network():
    input = keras.layers.Input(shape=X_shape, batch_size=batch_size)
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    output = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def epsilon_greedy(step, observation):
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        q_actions = mainQ.predict(np.expand_dims(observation, axis = 0))
        return np.argmax(q_actions)

def sample_expirience(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    return np.array(exp_buffer)[perm_batch]

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

@tf.function
def learn(source_states, actions, destination_states, rewards, dones):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4

    target_Q_values = targetQ(destination_states, training=False) # shape = (batch_size,4)
    target_y = rewards + discount_factor * tf.reduce_max(target_Q_values, axis=1) * (1 - dones) # shape = (batch_size,)
    
    with tf.GradientTape() as tape:
        pred_Q_values = mainQ(source_states, training=True) # shape = (batch_size,4)
        pred_y = tf.reduce_sum(tf.math.multiply(pred_Q_values, one_hot_actions_mask), axis=1) # Q values for non-chosen action do not impact loss. shape = (batch_size,)

        loss = mse_loss(pred_y, target_y)
    gradients = tape.gradient(loss, mainQ.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mainQ.trainable_variables))
    return loss

exp_buffer = deque(maxlen=80000)

mainQ = q_network()
targetQ = q_network()

np.random.random(0)
rewards_history = []

for i in range(num_episodes):
    done = False
    obs = env.reset()
    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    while not done:
        #env.render()
        chosen_action = epsilon_greedy(global_step, obs)
        next_obs, reward, done, _ = env.step(chosen_action)
        exp_buffer.append([tf.convert_to_tensor(obs, dtype=tf.float32),
                           chosen_action,
                           tf.convert_to_tensor(next_obs, dtype=tf.float32),
                           reward,
                           done])
        
        if global_step % steps_train == 0 and global_step > start_steps:
            samples = sample_expirience(batch_size)
            states_tensor = tf.stack(samples[:,0])
            actions_tensor = tf.convert_to_tensor(samples[:,1], dtype=tf.uint8)
            next_states_tensor = tf.stack(samples[:,2])
            rewards_tensor = tf.convert_to_tensor(samples[:,3], dtype=tf.float32)
            dones_tensor = tf.convert_to_tensor(samples[:,4], dtype=tf.float32)

            loss = learn(states_tensor,actions_tensor,next_states_tensor,rewards_tensor,dones_tensor)
            episodic_loss.append(loss)
            epsilon_decay()

        if (global_step + 1) % copy_step == 0 and global_step > start_steps:
            targetQ.set_weights(mainQ.get_weights())
        obs = next_obs
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print('[epoch ',i,' (steps: ',epoch_steps,')] Avg loss: ',np.mean(episodic_loss) if len(episodic_loss) > 0 else 0, ' Total reward: ', episodic_reward, f'Epsilon: {epsilon:.4f} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
targetQ.save('lunar_doubleDQN.h5')
env.close()
input("training complete...")