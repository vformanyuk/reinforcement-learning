import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

'''
Try also:
1) N-step returns. Consider only last N steps of episode
2) Lambda returns. G(t) = R(t+1) + gamma*(1-lambda(t+1))*V(S[t+1]) + gamma * lambda(t+1)*G(t+1)

3) Weighted returns. G(0)=V(S[0]), G(t) = Pi(A|S)/Mu(A|S)*(R[t+1] + gamma*G(t+1)). Where Mu(A|S) - copy of Pi before episode start
    This works when Pi was modified during training episode, or if Pi might be modified by other networks (A3C)
'''

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 5000
actor_learning_rate = 0.001
critic_learning_rate = 0.0005
X_shape = (env.observation_space.shape[0])
gamma = 0.99

checkpoint_step = 500

outputs_count = env.action_space.n

checkpoint_file_name = 'll_ac_checkpoint.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

def policy_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    actions_layer = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=actions_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(None, X_shape))
    x = keras.layers.Dense(512, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

actor = policy_network()
critic = value_network()

@tf.function(experimental_relax_shapes=True)
def train_actor(states, actions, advantage):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32) # shape = len(actions), 4
    
    with tf.GradientTape() as tape:
        actions_logits = actor(states, training=True)
        actions_distribution = tf.nn.log_softmax(actions_logits)
        
        loss = tf.reduce_mean(-tf.reduce_sum(actions_distribution * one_hot_actions_mask, axis=1) * advantage)
    gradients = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return loss

def calculate_Q(rewards):
    Q_tensor = []
    Q_target = 0.
    rewards_max_idx = len(rewards) - 1
    for j in tf.range(rewards_max_idx, -1, delta = -1):
        Q_target = rewards[j] + gamma*Q_target
        Q_tensor.append(Q_target)
    Q_tensor.reverse() #Very important to reverse calculated Q values as they are calculated backwards
    return tf.convert_to_tensor(Q_tensor, dtype = tf.float32)

@tf.function(experimental_relax_shapes=True)
def train_critic(states, Q):
    with tf.GradientTape() as tape:
        values = critic(states, training=True)
        values = tf.squeeze(values)
        advantage = Q - values
        loss = mse_loss(Q, values)
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, advantage

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

    Q = calculate_Q(episod_rewards)
    critic_loss, adv = train_critic(states_tensor, Q)
    actor_loss = train_actor(states_tensor,actions_tensor,adv)
    loss = critic_loss.numpy() + actor_loss.numpy()
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

