import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from time import sleep

from APEX.APEX_Rank_Priority_MemoryBuffer import APEX_Rank_Priority_MemoryBuffer
from APEX.neural_networks import policy_network, critic_network
from APEX.dpg_actor_slim import RunActor
from APEX.dpg_learner import RunLearner

from multiprocessing import Process, Pipe, Value
from threading import Thread, Lock

if __name__ == '__main__':
    orchestrator_debug_mode = False

    networks_initialized = False

    def orchestrator_log(msg):
        if orchestrator_debug_mode:
            print(f'[Orchestrator] {msg}')

    def actor_cmd_processor(actor, critic, replay_buffer, cmd_pipe, actor_weight_pipes, replay_data_pipes, net_sync_obj, data_sync_obj):
        connection_alive = True
        while connection_alive:
            try:
                cmd = cmd_pipe.recv()
                orchestrator_log(f'Got actor {cmd[1]} command {cmd[0]}')
                if cmd[0] == 0:
                    with net_sync_obj:
                        actor_weight_pipes[cmd[1]][1].send([actor.get_weights(), critic.get_weights()])
                    orchestrator_log(f'Sent target weights for actor {cmd[1]}')
                    continue
                if cmd[0] == 1:
                    replay_data = replay_data_pipes[cmd[1]][0].recv()
                    with data_sync_obj:
                        for r in replay_data:
                            # state, action, next_state, reward, gamma_power, done, td_error
                            replay_buffer.store(r[0], r[1], r[2], r[3], r[4], r[5], r[6])
                    orchestrator_log(f'Got replay data from actor {cmd[1]}')
                    continue
            except EOFError:
                print("Connection closed.")
                connection_alive = False
            except OSError:
                print('Handle closed.')
                connection_alive = False

    def learner_cmd_processor(actor, critic, replay_buffer, cmd_pipe, learner_weights_pipe, replay_data_pipe, priorities_pipe, net_sync_obj, data_sync_obj):
        global networks_initialized
        connection_alive = True
        while connection_alive:
            try:
                cmd = cmd_pipe.recv()
                orchestrator_log(f'Got learner command {cmd}')
                if cmd == 0: # update target networks
                    weights = learner_weights_pipe.recv()
                    with net_sync_obj:
                        #target_q.set_weights(weights[0])
                        actor.set_weights(weights[0])
                        critic.set_weights(weights[1])
                        networks_initialized = True
                    orchestrator_log(f'Target networks are updated')
                    continue
                if cmd == 1: # fetch N batches of data
                    data = []
                    with data_sync_obj:
                        for _ in range(learner_prefetch_batches):
                            data.append([*replay_buffer()])
                    replay_data_pipe.send(data)
                    orchestrator_log(f'Sent {learner_prefetch_batches} batches of data to learner')
                    continue
                if cmd == 2: # update priorities
                    data = priorities_pipe.recv()
                    with data_sync_obj:
                        for r in data:
                            replay_buffer.update_priorities(r[0], r[1])
                    continue
            except EOFError:
                print("Connection closed.")
                connection_alive = False
            except OSError:
                print('Handle closed.')
                connection_alive = False

    # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    env = gym.make('LunarLanderContinuous-v2')

    learner_batch_size = 128
    learner_prefetch_batches = 16

    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    gamma = 0.98

    actors_count = 2

    exp_buffer = APEX_Rank_Priority_MemoryBuffer(1000000, learner_batch_size, env.observation_space.shape, env.action_space.shape)

    weights_sync = Lock()
    data_sync = Lock()

    actor_cmd_read_pipe, actor_cmd_write_pipe = Pipe(False)
    learner_cmd_read_pipe, learner_cmd_write_pipe = Pipe(False)

    learner_weights_read_pipe, learner_weights_write_pipe = Pipe(False)
    learner_priorities_read_pipe, learner_priorities_write_pipe = Pipe(False)
    learner_replay_data_read_pipe, learner_replay_data_write_pipe = Pipe(False)

    replay_data_distribution_pipes = []
    weights_distribution_pipes = []
    actor_processess = []

    critic_net = critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
    policy_net = policy_network((env.observation_space.shape[0]), env.action_space.shape[0])

    cancelation_token = Value('i', 0)
    training_active_flag = Value('i', 0)
    buffer_ready = Value('i', 0)

    # Agenda
    # 1. Init networks at learner
    # 2. Distribute target networks to actors
    # 3. Fill up replay buffer
    # 4. Start learning

    actor_cmd_processor_thread = Thread(target=actor_cmd_processor, args=(policy_net, critic_net, exp_buffer, \
                                                                    actor_cmd_read_pipe, weights_distribution_pipes, replay_data_distribution_pipes, \
                                                                    weights_sync, data_sync))
    actor_cmd_processor_thread.start()

    learner_cmd_processor_thread = Thread(target=learner_cmd_processor, args=(policy_net, critic_net, exp_buffer, \
                                                                    learner_cmd_read_pipe, learner_weights_read_pipe, learner_replay_data_write_pipe, learner_priorities_read_pipe, \
                                                                    weights_sync, data_sync))
    learner_cmd_processor_thread.start()

    # 1. Init networks at learner
    learner_process = Process(target=RunLearner, args=(learner_batch_size, gamma, actor_learning_rate, critic_learning_rate, \
                                    (env.observation_space.shape[0],), (env.action_space.shape[0],), \
                                    learner_cmd_write_pipe, learner_weights_write_pipe, learner_replay_data_read_pipe, learner_priorities_write_pipe, \
                                    cancelation_token, training_active_flag, buffer_ready))
    learner_process.start()

    while not networks_initialized:
        sleep(1)

    # 2. Distribute target networks to actors
    for i in range(actors_count):
        weights_read_pipe, weights_write_pipe = Pipe(False)
        weights_distribution_pipes.append((weights_read_pipe, weights_write_pipe))
        replay_data_read_pipe, replay_data_write_pipe = Pipe(False)
        replay_data_distribution_pipes.append((replay_data_read_pipe, replay_data_write_pipe))
        p = Process(target=RunActor, args=(i, gamma, \
                                           actor_cmd_write_pipe, weights_read_pipe, replay_data_write_pipe, \
                                           cancelation_token, training_active_flag))
        actor_processess.append(p)
        p.start()

    print("Awaiting buffer fill up")
    # 3. Fill up replay buffer
    while len(exp_buffer) < learner_batch_size * learner_prefetch_batches:
        sleep(1)

    # 4. Start learning
    buffer_ready.value = 1
    input("training networks.\nPress enter to finish\n\n")
    cancelation_token.value = 1

    actor_cmd_read_pipe.close()
    actor_cmd_write_pipe.close()
    
    actor_cmd_processor_thread.join()

    learner_cmd_read_pipe.close()
    learner_cmd_write_pipe.close()
    learner_replay_data_write_pipe.close()
    learner_replay_data_read_pipe.close()
    learner_priorities_read_pipe.close()
    learner_priorities_write_pipe.close()
    
    learner_cmd_processor_thread.join()

    for i in range(len(weights_distribution_pipes)):
        weights_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        replay_data_distribution_pipes[i][0].close()
        weights_distribution_pipes[i][1].close()
        actor_processess[i].join()
    learner_process.join()