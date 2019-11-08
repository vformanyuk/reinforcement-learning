'''
Prioritized expirience replay
https://arxiv.org/abs/1511.05952
'''

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from per.SumTree import SumTree

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 800
global_step = 0
copy_step = 100
steps_train = 4
start_steps = 1200

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 5e-5

priority_eps = 0.00001
priority_alpha = 0.6
priority_beta = 0.4
priority_beta_step = (1-priority_beta)/num_episodes
piority_max_prob = 1

learning_rate = 0.001
batch_size = 64
X_shape = (env.observation_space.shape[0])
discount_factor = 0.97

exp_buffer_capacity = 100000

outputs_count = env.action_space.n

optimizer = tf.keras.optimizers.Adam(learning_rate)

memory = SumTree(exp_buffer_capacity)

def q_network():
    input = keras.layers.Input(shape=X_shape, batch_size=batch_size)
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    vals = keras.layers.Dense(1, activation='linear')(x)
    advs = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=[advs, vals])
    return model

def epsilon_greedy(step, observation):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        advantages, _ = mainQ.predict(np.expand_dims(observation, axis = 0))
        return np.argmax(advantages)

def priority_sample(batch_size):
    batch = []
    idxs = []
    segment = memory.total() / batch_size
    priorities = []

    beta = np.min([1., priority_beta + priority_beta_step])

    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)

        s = np.random.uniform(a, b)
        (idx, p, data) = memory.get(s)
        priorities.append(p)
        batch.append(data)
        idxs.append(idx)

    sampling_probabilities = priorities / memory.total()
    is_weight = np.power(memory.n_entries * sampling_probabilities, -beta)
    is_weight /= is_weight.max()
    return np.array(batch), idxs, is_weight

@tf.function
def learn(source_states, actions, destination_states, rewards, dones, importance_weights):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4

    target_advs, target_values = targetQ(destination_states, training=False) # shape = (batch_size,4)
    target_q = tf.add(target_values, (target_advs - tf.reduce_mean(target_advs, axis=1, keepdims = True))) #Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))
    target_y = rewards + discount_factor * tf.reduce_max(target_q, axis=1) * (1 - dones) # shape = (batch_size,)
    
    with tf.GradientTape() as tape:
        pred_advs, pred_values = mainQ(source_states, training=True) # shape = (batch_size,4)
        pred_q = tf.add(pred_values, (pred_advs - tf.reduce_mean(pred_advs, axis=1, keepdims = True))) #Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))
        pred_y = tf.reduce_sum(tf.math.multiply(pred_q, one_hot_actions_mask), axis=1) # Q values for non-chosen action do not impact loss. shape = (batch_size,)
        
        td_error = tf.math.subtract(target_y, pred_y)
        loss = tf.reduce_mean(tf.math.multiply(tf.math.square(td_error), importance_weights))
    gradients = tape.gradient(loss, mainQ.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mainQ.trainable_variables))
    return loss, tf.math.abs(td_error)

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

def get_priority(err):
    return np.power(err + priority_eps, priority_alpha)

mainQ = q_network()
targetQ = q_network()

np.random.random()
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
        memory.add(get_priority(piority_max_prob), [tf.convert_to_tensor(obs, dtype=tf.float32),
                           chosen_action,
                           tf.convert_to_tensor(next_obs, dtype=tf.float32),
                           reward,
                           float(done)])
        
        if global_step > start_steps and global_step % steps_train == 0:
            samples, sample_idxs, is_weights = priority_sample(batch_size)
            states_tensor = tf.stack(samples[:,0])
            actions_tensor = tf.convert_to_tensor(samples[:,1], dtype=tf.uint8)
            next_states_tensor = tf.stack(samples[:,2])
            rewards_tensor = tf.convert_to_tensor(samples[:,3], dtype=tf.float32)
            dones_tensor = tf.convert_to_tensor(samples[:,4], dtype=tf.float32)
            is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float32)

            loss, td_errors = learn(states_tensor,actions_tensor,next_states_tensor,rewards_tensor,dones_tensor,is_weights_tensor)
            episodic_loss.append(loss)
            epsilon_decay()

            #update priorities
            for batch_idx, err in enumerate(td_errors):
                memory.update(sample_idxs[batch_idx], get_priority(err))
            piority_max_prob = max(td_errors)

        if (global_step + 1) % copy_step == 0 and global_step > start_steps:
            targetQ.set_weights(mainQ.get_weights())
        obs = next_obs
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    priority_beta += priority_beta_step
    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[:-100])

    print('[epoch ',i,' (steps: ',epoch_steps,')] Avg loss: ',np.mean(episodic_loss) if len(episodic_loss) > 0 else 0, ' Total reward: ', episodic_reward, f'Epsilon: {epsilon:.4f} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
targetQ.save('lunar_dueling_ddqn_IS.h5')
env.close()
input("training complete...")