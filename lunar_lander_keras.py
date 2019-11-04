import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

env = gym.make('LunarLander-v2')

num_episodes = 800
learning_rate = 0.001
batch_size = 48
X_shape = (env.observation_space.shape[0])
discount_factor = 0.98

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 1/(2.5*num_episodes)

global_step = 0
copy_step = 50
steps_train = 4
start_steps = 1000

outputs_count = env.action_space.n

exp_buffer = deque(maxlen=80000)

optimizer = tf.keras.optimizers.Adam(learning_rate)

def q_network():
    model = keras.Sequential([keras.layers.Input(shape=X_shape, batch_size=batch_size),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(outputs_count, activation='linear')])
    return model

def epsilon_greedy(step, observation):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        q_actions = mainQ.predict(np.expand_dims(observation, axis = 0))
        return np.argmax(q_actions)

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

def sample_expirience(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    return np.array(exp_buffer)[perm_batch]

def learn(source_states, actions, destination_states, rewards, dones):
    pred_Q_values = mainQ.predict(source_states) # shape = (48,4)
    target_Q_values = targetQ.predict(destination_states) # shape = (48,4)

    target_y = np.array(pred_Q_values[:,:])
    
    idxs = np.arange(batch_size)
    target_y[idxs,actions] = rewards + discount_factor * np.max(target_Q_values, axis=1) * (1 - dones)

    loss = mainQ.train_on_batch(source_states,target_y)
    return loss

np.random.random()

mainQ = q_network()
mainQ.compile(optimizer=optimizer, loss='mean_squared_error')

targetQ = q_network()

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
        exp_buffer.append([obs,
                           chosen_action,
                           next_obs,
                           reward,
                           done])
        
        if global_step % steps_train == 0 and global_step > start_steps:
            samples = sample_expirience(batch_size)
            states_tensor = np.stack(samples[:,0])
            actions_tensor = samples[:,1].astype(np.int32)
            next_states_tensor = np.stack(samples[:,2])
            rewards_tensor = samples[:,3]
            dones_tensor = samples[:,4].astype(np.float32)

            loss = learn(states_tensor,actions_tensor,next_states_tensor,rewards_tensor,dones_tensor)
            episodic_loss.append(loss)
            epsilon_decay()

        if (global_step + 1) % copy_step == 0 and global_step > start_steps:
            targetQ.set_weights(mainQ.get_weights())
        obs = next_obs
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward
    print('[epoch ',i,' (steps per epoch: ',epoch_steps,')] Avg loss: ',np.mean(episodic_loss) if len(episodic_loss) > 0 else 0, ' Total reward: ', episodic_reward, f'Epsilon: {epsilon:.4f}')
targetQ.save('D:\Projects\RL\RL\lunar.h5')
env.close()