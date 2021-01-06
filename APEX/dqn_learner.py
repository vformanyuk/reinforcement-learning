import gym
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from time import sleep
from typing import Tuple

from APEX.APEX_Rank_Priority_MemoryBuffer import APEX_Rank_Priority_MemoryBuffer
from APEX.neural_networks import q_network

CMD_SET_NETWORK_WEIGHTS = 0
CMD_GET_REPLAY_DATA = 1
CMD_UPDATE_PRIORITIES = 2

class Learner(object):
    def __init__(self, batch_size:float, gamma:float, learning_rate:float,
                 state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...],
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                 cancellation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value,
                 *args, **kwargs):
        self.cancellation_token = cancellation_token
        self.training_active = training_active_flag
        self.buffer_ready_flag = buffer_ready

        # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.01

        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_data_pipe = replay_data_pipe
        self.priorities_pipe = priorities_pipe

        RND_SEED = 0x12345
        tf.random.set_seed(RND_SEED)
        np.random.random(RND_SEED)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.action_space_dims = action_space_shape[0]

        self.main_Q = q_network(state_space_shape[0], action_space_shape[0])
        self.target_Q = q_network(state_space_shape[0], action_space_shape[0])
        self.target_Q.set_weights(self.main_Q.get_weights())

    def validate(self):
        env = gym.make('LunarLander-v2')
        done = False
        observation = env.reset()

        episodic_reward = 0

        while not done:
            #env.render()
            q_actions =  self.main_Q(np.expand_dims(observation, axis = 0), training=False)[0].numpy()
            next_observation, reward, done, _ = env.step(np.argmax(q_actions))
            observation = next_observation
            episodic_reward += reward
        env.close()
        print(f'\t\t[Learner] Validation run total reward = {episodic_reward}')

    def run(self):
        self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS) #initial target networks distribution
        self.weights_pipe.send([self.main_Q.get_weights()])

        while self.buffer_ready_flag.value < 1:
            sleep(1)

        training_runs = 1
        while self.cancellation_token.value == 0:
            self.cmd_pipe.send(CMD_GET_REPLAY_DATA)
            batches = self.replay_data_pipe.recv()
            
            priorities_updates = []
            for b in batches:
                loss, td_errors = self.__train(b[0],b[1],b[2],b[3],b[4],b[5],b[6])
                priorities_updates.append((b[7], td_errors))
            
            self.cmd_pipe.send(CMD_UPDATE_PRIORITIES)
            self.priorities_pipe.send(priorities_updates)
            
            if self.training_active.value == 0:
                self.training_active.value = 1
            if training_runs % 10 == 0:
                self.__soft_update_models()
            if training_runs % 25 == 0:
                self.validate()
            training_runs += 1

    @tf.function
    def __train(self, states, actions, next_states, rewards, gamma_powers, dones, is_weights):
        one_hot_actions_mask = tf.one_hot(actions, depth=self.action_space_dims, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4

        target_Q_values = self.target_Q(next_states, training=False) # shape = (batch_size,4)
        target_y = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * tf.reduce_max(target_Q_values, axis=1) * (1 - dones) # shape = (batch_size,)
    
        with tf.GradientTape() as tape:
            pred_Q_values = self.main_Q(states, training=True) # shape = (batch_size,4)
            pred_y = tf.reduce_sum(tf.math.multiply(pred_Q_values, one_hot_actions_mask), axis=1) # Q values for non-chosen action do not impact loss. shape = (batch_size,)

            td_error = target_y - pred_y
            loss = tf.reduce_mean(tf.math.pow(td_error, 2))
        gradients = tape.gradient(loss, self.main_Q.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_Q.trainable_variables))
        return loss, tf.math.abs(td_error)

    def __soft_update_models(self):
        target_weights = self.target_Q.get_weights()
        q_weights = self.main_Q.get_weights()
        updated_weights = []
        for qw,tqw in zip(q_weights,target_weights):
            updated_weights.append(self.tau * qw + (1.0 - self.tau) * tqw)
        self.target_Q.set_weights(updated_weights)

def RunLearner(batch_size:int, gamma:float, leraning_rate:float,
               state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...],
                cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                cancelation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value):
    learner = Learner(batch_size, gamma, leraning_rate, 
                      state_space_shape, action_space_shape,
                      cmd_pipe, weights_pipe, replay_data_pipe, priorities_pipe, 
                      cancelation_token, training_active_flag, buffer_ready)
    learner.run()
