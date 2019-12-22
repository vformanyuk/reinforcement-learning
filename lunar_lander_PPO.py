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
actor_learning_rate = 0.001
critic_learning_rate = 0.0005
clipping_epsilon = 0.2
batch_size = 64
X_shape = (env.observation_space.shape[0])
gamma = 0.99
gae_lambda = 0.95
entropy_beta = 0.01

lambda_gamma_constant = tf.constant(gae_lambda * gamma, dtype=tf.float32)

checkpoint_step = 500

outputs_count = env.action_space.n

actor_checkpoint_file_name = 'll_ppo_actor_checkpoint.h5'
critic_checkpoint_file_name = 'll_ppo_critic_checkpoint.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

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
    x = keras.layers.Dense(128, activation='relu')(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

@tf.function(experimental_relax_shapes=True)
def train_actor(states, actions, target_distributions, adv):
    one_hot_actions_mask = tf.one_hot(actions, depth=outputs_count, on_value = 1.0, off_value = 0.0, dtype=tf.float32)

    with tf.GradientTape() as tape:
        action_logits = tf.squeeze(evaluation_policy(states, training=True))
        evalution_distribution = tf.nn.softmax(action_logits)

        with tape.stop_recording():
            evalution_log_distribution = tf.nn.log_softmax(action_logits)
            entropy = -tf.reduce_sum(evalution_log_distribution * evalution_distribution)

        r = tf.reduce_sum(evalution_distribution * one_hot_actions_mask, axis=1) / target_distributions
        r_clipped = tf.clip_by_value(r, 1 - clipping_epsilon, 1 + clipping_epsilon)
        loss = -tf.reduce_mean(tf.math.minimum(r * adv, r_clipped * adv)) + entropy_beta * entropy
    gradients = tape.gradient(loss, evaluation_policy.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, evaluation_policy.trainable_variables))
    return loss

#@tf.function(experimental_relax_shapes=True)
def train_critic(states, rewards, trajectory_len):
    V_target = tf.TensorArray(dtype = tf.float32, size = trajectory_len)
    V_target_idx = 0
    
    gae_tensor = tf.TensorArray(dtype = tf.float32, size = trajectory_len)
    gae_tensor_idx = trajectory_len - 1
    gae = 0.
    gae_power = 0.
    
    end_idx = len(rewards) - 1
    start_idx = end_idx - trajectory_len

    with tf.GradientTape() as tape:
        V = critic(states, training=True)

        for j in tf.range(end_idx, start_idx, delta = -1):
            V_next = V[j+1] if (j+1) <= end_idx else tf.constant(0., dtype=tf.float32, shape=(1,))
            delta = rewards[j] + gamma * V_next
            V_target.write(V_target_idx, delta)
            gae += tf.math.pow(lambda_gamma_constant, gae_power) * tf.squeeze(delta - V[j])
            gae_tensor.write(gae_tensor_idx, gae)
            
            gae_tensor_idx -= 1 #filling tensor array from behind, so no need to reverse
            V_target_idx += 1
            gae_power += 1
        advantage = gae_tensor.stack()
        if len(advantage) > 1:
            advantage = (advantage - tf.reduce_mean(advantage)) / tf.math.reduce_std(advantage) # for shape(1,1) return is NaN

        loss = mse_loss(V_target.stack(), V)
        #loss = tf.reduce_mean(tf.square(advantage))
    gradients = tape.gradient(loss, critic.trainable_variables) #switching to tf varibles broke gradients
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, advantage

'''
GAE(t,L,lambda,gamma) = sum((lambda*gamma)^l*delta(t+l), l = [0...L])
delta(t) = R(t) + V(t+1) - V(t)
'''
def calculate_GAE(rewards, V, isTerminal):
    gae_tensor=[]
    gae = 0
    lambda_gamma_multiplier = 1
    end_idx = len(rewards) - 1

    if not isTerminal:
        end_idx -= 1 # because V[j+1] does not exist for edge case - shift interval back by 1
    else:
        V.append(0) # for terminal state there are no future rewards

    start_idx = end_idx - gae_minibatch_size - 1 # tf.range end index(start_idx in this case) is not inclusive
    for j in tf.range(end_idx, start_idx, delta = -1):
        delta = (rewards[j] + gamma * V[j+1]) - V[j]
        gae += (lambda_gamma_multiplier * delta)
        lambda_gamma_multiplier *= (gae_lambda * gamma)
        gae_tensor.append(gae)
    #gae_tensor.reverse() #don't reverse gae tensor. Reversed order is required to update adv in replay buffer, MSE doesn't care about ordering
    gae_tensor = (gae_tensor - np.mean(gae_tensor)) / np.std(gae_tensor)
    return gae_tensor

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

tf.random.set_seed(0x12345)
np.random.random(0)

rewards_history = []
copy_batch_step = 0

for i in range(num_episodes):
    done = False
    observation = env.reset()
    critic_loss_history = []
    actor_loss_history = []

    episod_rewards = []
    states_memory = []
    actions_memory = []
    action_prob_memory = []
    epoch_steps = 0
    processed_batches = 0

    while not done:
        #env.render()
        actions_logits = target_policy(np.expand_dims(observation, axis = 0), training=False)
        actions_logits = tf.squeeze(actions_logits)
        actions_distribution = tf.nn.softmax(actions_logits).numpy()

        chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
        next_observation, reward, done, _ = env.step(chosen_action)

        episod_rewards.append(reward)
        states_memory.append(tf.convert_to_tensor(observation, dtype=tf.float32))
        actions_memory.append(chosen_action)
        action_prob_memory.append(actions_distribution[chosen_action])
        epoch_steps += 1

        # obtain trajectory segment and train networks
        if (epoch_steps > 0 and epoch_steps % batch_size == 0) or done:
            trajectory_length = batch_size
            if done:
                trajectory_length = epoch_steps - processed_batches * batch_size
            critic_loss, adv = train_critic(tf.stack(states_memory[-trajectory_length:]),
                                            tf.convert_to_tensor(episod_rewards[-trajectory_length:], dtype=tf.float32), 
                                            tf.convert_to_tensor(trajectory_length, dtype=tf.int32))
            critic_loss_history.append(critic_loss)

            actor_loss = train_actor(tf.stack(states_memory[-trajectory_length:]),
                                     tf.convert_to_tensor(actions_memory[-trajectory_length:], dtype=tf.int32), 
                                     tf.convert_to_tensor(action_prob_memory[-trajectory_length:], dtype=tf.float32),
                                     adv)
            actor_loss_history.append(actor_loss)

            processed_batches += 1
            copy_batch_step += 1

        #update target policy every 4th batch or in the end of episode
        if (copy_batch_step > 0 and copy_batch_step % 3 == 0) or done:
            copy_batch_step = 0
            target_policy.set_weights(evaluation_policy.get_weights())

        observation = next_observation

    #if i % checkpoint_step == 0 and i > 0:
    #    policy.save(checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor mloss: {np.mean(actor_loss_history):.4f} Critic mloss: {np.mean(critic_loss_history):.4f} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
policy.save('lunar_policy_grad.h5')
input("training complete...")
