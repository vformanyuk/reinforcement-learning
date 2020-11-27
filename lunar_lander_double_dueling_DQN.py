import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl_utils.SARST_RandomAccess_MemoryBuffer import SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 1000
global_step = 0
steps_train = 4
copy_step = 100
start_steps = 1200

epsilon = 1
epsilon_min = 0.01
epsilon_decay_steps = 1.5e-4

learning_rate = 3e-4
batch_size = 128
X_shape = (env.observation_space.shape[0])
discount_factor = 0.98

exp_buffer_capacity = 100000

outputs_count = env.action_space.n

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

optimizer = tf.keras.optimizers.Adam(learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

def q_network():
    input = keras.layers.Input(shape=X_shape, batch_size=batch_size)
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    vals = keras.layers.Dense(1, activation='linear')(x)
    advs = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=[advs, vals])
    return model

def epsilon_greedy(observation):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        advantages, _ = mainQ.predict(np.expand_dims(observation, axis = 0))
        return np.argmax(advantages)

@tf.function
def learn(source_states, actions, destination_states, rewards, dones):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4

    target_advs, target_values = targetQ(destination_states, training=False)
    target_q = tf.add(target_values, (target_advs - tf.reduce_mean(target_advs, axis=1, keepdims = True))) #Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))
    target_y = rewards + discount_factor * tf.reduce_max(target_q, axis=1) * (1 - dones) # shape = (batch_size,)
    
    with tf.GradientTape() as tape:
        pred_advs, pred_values = mainQ(source_states, training=True)
        
        #with tape.stop_recording():
        pred_q = tf.add(pred_values, (pred_advs - tf.reduce_mean(pred_advs, axis=1, keepdims = True))) #Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))
        pred_y = tf.reduce_sum(tf.math.multiply(pred_q, one_hot_actions_mask), axis=1) # Q values for non-chosen action do not impact loss. shape = (batch_size,)
        
        loss = mse_loss(target_y,pred_y)
    gradients = tape.gradient(loss, mainQ.trainable_weights)
    optimizer.apply_gradients(zip(gradients, mainQ.trainable_weights))
    return loss

def epsilon_decay():
    global epsilon
    epsilon = epsilon - epsilon_decay_steps if epsilon > epsilon_min else epsilon_min

exp_buffer = SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape, action_type=np.int32)

mainQ = q_network()
targetQ = q_network()

rewards_history = []

for i in range(num_episodes):
    done = False
    obs = env.reset()
    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    while not done:
        #env.render()
        chosen_action = epsilon_greedy(obs)
        next_obs, reward, done, _ = env.step(chosen_action)
        exp_buffer.store(obs, chosen_action, next_obs, reward, float(done))
        
        if global_step > start_steps and global_step % steps_train == 0:
            states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor = exp_buffer(batch_size)

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
    print(f'[epoch {i} ({epoch_steps})] Avg loss: {np.mean(episodic_loss):.4f} Epsilon: {epsilon:.4f} Total reward: {episodic_reward:.4f} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    targetQ.save('lunar_dueling_ddqn.h5')
env.close()
input("training complete...")