import gym
import os
import numpy as np
import tensorflow as tf
from time import sleep

from R2D2.R2D2_TrajectoryStore import R2D2_TrajectoryStore
from R2D2.neural_networks import policy_network, critic_network
from R2D2.R2D2_SAC_Agent import RunActor
from R2D2.R2D2_SAC_Learner import RunLearner
from R2D2.DTOs import AgentTransmitionBuffer, LearnerTransmitionBuffer

from multiprocessing import Process, Pipe, Value
from threading import Thread, Lock

LEARNER_CMD_SET_NETWORK_WEIGHTS = 0
LEARNER_CMD_GET_REPLAY_DATA = 1
LEARNER_CMD_UPDATE_PRIORITIES = 2

ACTOR_CMD_GET_NETWORKS = 0
ACTOR_CMD_SEND_REPLAY_DATA = 1

if __name__ == '__main__':
    orchestrator_debug_mode = False
    networks_initialized = False

    def orchestrator_log(msg):
        if orchestrator_debug_mode:
            print(f'[Orchestrator ({os.getpid()})] {msg}')

    def actor_cmd_processor(actor, critic1, critic2, replay_buffer:R2D2_TrajectoryStore, \
                            cmd_pipe, actor_weight_pipes, replay_data_pipes, net_sync_obj, data_sync_obj):
        global alpha_log
        connection_alive = True
        while connection_alive:
            try:
                cmd = cmd_pipe.recv()
                orchestrator_log(f'Got actor {cmd[1]} command {cmd[0]}')
                if cmd[0] == ACTOR_CMD_GET_NETWORKS: # actor requested networks update
                    with net_sync_obj:
                        actor_weight_pipes[cmd[1]][1].send([actor.get_weights(), critic1.get_weights(), critic2.get_weights()])
                    orchestrator_log(f'Sent target weights for actor {cmd[1]}')
                    continue
                if cmd[0] == ACTOR_CMD_SEND_REPLAY_DATA: # actor sends replay data
                    replay_data:AgentTransmitionBuffer = replay_data_pipes[cmd[1]][0].recv() # AgentTransmitionBuffer recieved 
                    with data_sync_obj:
                        for actor_hidden_state, burn_in, states, actions, next_states, rewards, gammas, dones, td_error in replay_data:
                            # store whole trajectory along with burn-in, actor hidden state and td_error
                            replay_buffer.store(actor_hidden_state, burn_in, [states, actions, next_states, rewards, gammas, dones], len(rewards), td_error)
                    orchestrator_log(f'Got replay data from actor {cmd[1]}')
                    continue
            except EOFError:
                print("Connection closed.")
                connection_alive = False
            except OSError:
                print('Handle closed.')
                connection_alive = False

    def learner_cmd_processor(actor, critic1, critic2, replay_buffer:R2D2_TrajectoryStore, \
                             cmd_pipe, learner_weights_pipe, replay_data_pipe, priorities_pipe, net_sync_obj, data_sync_obj):
        global networks_initialized
        global alpha_log
        connection_alive = True
        while connection_alive:
            try:
                cmd = cmd_pipe.recv()
                orchestrator_log(f'Got learner command {cmd}')
                if cmd == LEARNER_CMD_SET_NETWORK_WEIGHTS: # update target networks
                    weights = learner_weights_pipe.recv()
                    with net_sync_obj:
                        actor.set_weights(weights[0])
                        critic1.set_weights(weights[1])
                        critic2.set_weights(weights[2])
                        networks_initialized = True
                    orchestrator_log(f'Target networks are updated')
                    continue
                if cmd == LEARNER_CMD_GET_REPLAY_DATA: # fetch mini batch of trajectories for learner
                    data = LearnerTransmitionBuffer()
                    with data_sync_obj:
                        for actor_hidden_state, burn_in, trajectory, is_weights, meta_idx in replay_buffer.sample(trajectories_mini_batch):
                            data.append(actor_hidden_state, 
                                        burn_in, 
                                        trajectory[0], 
                                        trajectory[1], 
                                        trajectory[2], 
                                        trajectory[3], 
                                        trajectory[4], 
                                        trajectory[5], 
                                        is_weights,
                                        meta_idx)
                    replay_data_pipe.send(data)
                    orchestrator_log(f'Sent {trajectories_mini_batch} batches of data to learner')
                    continue
                if cmd == LEARNER_CMD_UPDATE_PRIORITIES: # update priorities
                    data = priorities_pipe.recv()
                    with data_sync_obj:
                        replay_buffer.update_priorities(data[0], data[1])
                    orchestrator_log(f'Updated trajectory priorities recieved')
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

    trajectories_mini_batch = 64

    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    gamma = 0.99

    actors_count = 2
    
    stack_size = 4
    state_space_shape = (stack_size, env.observation_space.shape[0])
    outputs_count = env.action_space.shape[0]
    actor_recurrent_layer_size = 256

    trajectory_length = 100
    burn_in_length = 28

    exp_buffer = R2D2_TrajectoryStore(buffer_size=1000000, alpha=0.7, beta=0.5, beta_increase_rate=1)

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

    critic1_net = critic_network(state_space_shape, outputs_count, actor_recurrent_layer_size)
    critic2_net = critic_network(state_space_shape, outputs_count, actor_recurrent_layer_size)
    policy_net = policy_network(state_space_shape, outputs_count, actor_recurrent_layer_size)

    cancelation_token = Value('i', 0)
    training_active_flag = Value('i', 0)
    buffer_ready = Value('i', 0)

    # Agenda
    # 1. Init networks at learner
    # 2. Distribute target networks to actors
    # 3. Fill up replay buffer
    # 4. Start learning

    actor_cmd_processor_thread = Thread(target=actor_cmd_processor, args=(policy_net, critic1_net, critic2_net, exp_buffer, \
                                                                    actor_cmd_read_pipe, weights_distribution_pipes, replay_data_distribution_pipes, \
                                                                    weights_sync, data_sync))
    actor_cmd_processor_thread.start()

    learner_cmd_processor_thread = Thread(target=learner_cmd_processor, args=(policy_net, critic1_net, critic2_net, exp_buffer, \
                                                                    learner_cmd_read_pipe, learner_weights_read_pipe, learner_replay_data_write_pipe, learner_priorities_read_pipe, \
                                                                    weights_sync, data_sync))
    learner_cmd_processor_thread.start()

    # 1. Init networks at learner
    learner_process = Process(target=RunLearner, args=(trajectories_mini_batch, gamma, actor_learning_rate, critic_learning_rate, \
                                    state_space_shape, (outputs_count,), actor_recurrent_layer_size, \
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

    orchestrator_log("Awaiting buffer fill up")
    # 3. Fill up replay buffer
    while len(exp_buffer) < 10 * trajectories_mini_batch:
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