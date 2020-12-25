import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from APEX.APEX_Rank_Priority_MemoryBuffer import APEX_Rank_Priority_MemoryBuffer
from rl_utils.OUActionNoise import OUActionNoise
from APEX.neural_networks import policy_network, critic_network
from APEX.dpg_actor import RunActor
from multiprocessing import Process, Pipe, Value
from threading import Thread

if __name__ == '__main__':
    orchestrator_debug_mode = False

def orchestrator_log(msg):
    if orchestrator_debug_mode:
        print(f'[Orchestrator] {msg}')

def cmd_processor(actor, critic, replay_buffer, cmd_pipe, actor_weight_pipes, replay_data_pipes):
    connection_alive = True
    while connection_alive:
        try:
            cmd = cmd_pipe.recv()
            orchestrator_log(f'Got actor {cmd[1]} command {cmd[0]}')
            if cmd[0] == 0:
                actor_weight_pipes[cmd[1]][1].send([actor.get_weights(), critic.get_weights()])
                orchestrator_log(f'Sent target weights for actor {cmd[1]}')
                continue
            if cmd[0] == 1:
                replay_data = replay_data_pipes[cmd[1]][0].recv()
                for r in replay_data:
                    # state, action, next_state, reward, gamma_power, done, td_error
                    replay_buffer.store(r[0], r[1], r[2], r[3], r[4], r[5], r[6])
                orchestrator_log(f'Got replay data from actor {cmd[1]}')
                continue
        except EOFError:
            print("Connection closed.")
            connection_alive = False
        except OSError:
            print("Handle closed")
            connection_alive = False

def stop_training(token:Value):
    input("training networks.\nPress enter to finish\n\n")
    token.value = 1

def soft_update_models(actor, target_policy, critic, target_critic, tau):
    target_actor_weights = target_policy.get_weights()
    actor_weights = actor.get_weights()
    updated_actor_weights = []
    for aw,taw in zip(actor_weights,target_actor_weights):
        updated_actor_weights.append(tau * aw + (1.0 - tau) * taw)
    target_policy.set_weights(updated_actor_weights)

    target_critic_weights = target_critic.get_weights()
    critic_weights = critic.get_weights()
    updated_critic_weights = []
    for cw,tcw in zip(critic_weights,target_critic_weights):
        updated_critic_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic.set_weights(updated_critic_weights)

if __name__ == '__main__':
    # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    env = gym.make('LunarLanderContinuous-v2')

    learner_batch_size = 128
    actor_batch_size = 64
    num_episodes = 5000
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    gamma = 0.99
    tau = 0.001

    RND_SEED = 0x12345

    actors_count = 2

    actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    tf.random.set_seed(RND_SEED)
    np.random.random(RND_SEED)

    exp_buffer_capacity = 1000000
    exp_buffer = APEX_Rank_Priority_MemoryBuffer(exp_buffer_capacity, learner_batch_size, env.observation_space.shape, env.action_space.shape)

    cmd_read_pipe, cmd_write_pipe = Pipe(False)

    replay_data_distribution_pipes = []
    weights_distribution_pipes = []
    actor_processess = []

    actor_net = policy_network((env.observation_space.shape[0]), env.action_space.shape[0])
    target_actor_net = policy_network((env.observation_space.shape[0]), env.action_space.shape[0])
    target_actor_net.set_weights(actor_net.get_weights())

    critic_net = critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
    target_critic_net = critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
    target_critic_net.set_weights(critic_net.get_weights())

    cancelation_token = Value('i', 0)
    training_active_flag = Value('i', 0)

    cmd_processor_thread = Thread(target=cmd_processor, args=(actor_net, critic_net, exp_buffer, cmd_read_pipe, weights_distribution_pipes, replay_data_distribution_pipes))
    cmd_processor_thread.start()

    @tf.function
    def train_actor_critic(states, actions, next_states, rewards, gamma_powers, dones, is_weights):
        target_mu = target_actor_net(next_states, training=False)
        target_q = rewards + tf.math.pow(gamma, gamma_powers + 1) * tf.reduce_sum((1 - dones) * target_critic_net([next_states, target_mu], training=False), axis = 1)

        with tf.GradientTape() as tape:
            current_q = critic_net([states, actions], training=True)
            #c_loss = mse_loss(current_q, target_q)
            td_errors = target_q - tf.squeeze(current_q, axis=1)
            c_loss = tf.reduce_mean(is_weights * tf.math.pow(td_errors, 2), axis=0)
        gradients = tape.gradient(c_loss, critic_net.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients, critic_net.trainable_variables))

        with tf.GradientTape() as tape:
            current_mu = actor_net(states, training=True)
            current_q = critic_net([states, current_mu], training=False)
            a_loss = tf.reduce_mean(-current_q)
        gradients = tape.gradient(a_loss, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(gradients, actor_net.trainable_variables))
        return a_loss, c_loss, td_errors

    for i in range(actors_count):
        weights_read_pipe, weights_write_pipe = Pipe(False)
        weights_distribution_pipes.append((weights_read_pipe, weights_write_pipe))
        replay_data_read_pipe, replay_data_write_pipe = Pipe(False)
        replay_data_distribution_pipes.append((replay_data_read_pipe, replay_data_write_pipe))
        p = Process(target=RunActor, args=(i, actor_batch_size, gamma, actor_learning_rate, critic_learning_rate, \
                                           cmd_write_pipe, weights_read_pipe, replay_data_write_pipe, cancelation_token, training_active_flag))
        actor_processess.append(p)
        p.start()

    halt_thread = Thread(target=stop_training, args=(cancelation_token,))
    halt_thread.start()

    soft_update_counter = 0
    while cancelation_token.value == 0:
        if len(exp_buffer) > 10 * learner_batch_size:
            training_active_flag.value = 1
            states_tensor, actions_tensor, next_states_tensor, rewards_tensor, gamma_powers_tensor, dones_tensor, is_weights_tensor, meta_idxs = exp_buffer()
            actor_loss, critic_loss, td_errors = train_actor_critic(states_tensor, actions_tensor, next_states_tensor, rewards_tensor, gamma_powers_tensor, dones_tensor, is_weights_tensor)
            exp_buffer.update_priorities(meta_idxs, td_errors)
            if soft_update_counter > 64:
                soft_update_models(actor_net, target_actor_net, critic_net, target_critic_net, tau)
                soft_update_counter = 0
            soft_update_counter += 1

            #actor_loss_history.append(actor_loss)
            #critic_loss_history.append(critic_loss)
            
    cmd_read_pipe.close()
    cmd_processor_thread.join()
    halt_thread.join()

    cmd_write_pipe.close()
    for i in range(len(weights_distribution_pipes)):
        weights_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        replay_data_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        actor_processess[i].join()