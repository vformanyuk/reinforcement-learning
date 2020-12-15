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
                    # state, action, next_state, reward, done, td_error
                    replay_buffer.store(r[0], r[1], r[2], r[3], r[4], r[5])
                orchestrator_log(f'Got replay data from actor {cmd[1]}')
                continue
        except EOFError:
            print("Connection closed.")
            connection_alive = False
        except OSError:
            print("Handle closed")
            connection_alive = False

if __name__ == '__main__':
    # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    env = gym.make('LunarLanderContinuous-v2')

    batch_size = 64
    num_episodes = 5000
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    gamma = 0.99
    tau = 0.001

    RND_SEED = 0x12345

    actors_count = 4

    actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    tf.random.set_seed(RND_SEED)
    np.random.random(RND_SEED)

    exp_buffer_capacity = 1000000
    exp_buffer = APEX_Rank_Priority_MemoryBuffer(exp_buffer_capacity, batch_size, env.observation_space.shape, env.action_space.shape)

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

    cmd_processor_thread = Thread(target=cmd_processor, args=(actor_net, critic_net, exp_buffer, cmd_read_pipe, weights_distribution_pipes, replay_data_distribution_pipes))
    cmd_processor_thread.start()

    for i in range(actors_count):
        weights_read_pipe, weights_write_pipe = Pipe(False)
        weights_distribution_pipes.append((weights_read_pipe, weights_write_pipe))
        replay_data_read_pipe, replay_data_write_pipe = Pipe(False)
        replay_data_distribution_pipes.append((replay_data_read_pipe, replay_data_write_pipe))
        p = Process(target=RunActor, args=(i, batch_size, gamma, actor_learning_rate, critic_learning_rate, \
                                           cmd_write_pipe, weights_read_pipe, replay_data_write_pipe, cancelation_token))
        actor_processess.append(p)
        p.start()

    # Train networks here (Learner)

    input("training networks.\nPress enter to finish\n\n")
    cancelation_token.value = 1
    cmd_read_pipe.close()
    cmd_processor_thread.join()

    cmd_write_pipe.close()
    for i in range(len(weights_distribution_pipes)):
        weights_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        replay_data_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        actor_processess[i].join()