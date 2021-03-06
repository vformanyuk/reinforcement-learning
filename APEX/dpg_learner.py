import gym
import numpy as np
import os
import tensorflow as tf
import multiprocessing as mp

from tensorflow import keras
from time import sleep
from typing import Tuple

from APEX.APEX_Rank_Priority_MemoryBuffer import APEX_Rank_Priority_MemoryBuffer
from APEX.neural_networks import policy_network, critic_network

CMD_SET_NETWORK_WEIGHTS = 0
CMD_GET_REPLAY_DATA = 1
CMD_UPDATE_PRIORITIES = 2

class Learner(object):
    def __init__(self, batch_size:float, gamma:float, actor_learning_rate:float, critic_learning_rate:float, 
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
        self.finish_criteria = 200
        self.checkpoint_step = 100

        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_data_pipe = replay_data_pipe
        self.priorities_pipe = priorities_pipe

        RND_SEED = 0x12345
        tf.random.set_seed(RND_SEED)
        np.random.random(RND_SEED)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

        self.actor_network_file = "apex-dpg-learner-actor.h5"
        self.target_actor_network_file = "apex-dpg-learner-target_actor.h5"
        self.critic_network_file = "apex-dpg-learner-critic.h5"
        self.target_critic_network_file = "apex-dpg-learner-target_critic.h5"

        if os.path.isfile(self.actor_network_file):
            self.actor = keras.models.load_model(self.actor_network_file)
            print("Actor Model restored from checkpoint.")
        else:
            self.actor = policy_network(state_space_shape[0], action_space_shape[0])
        if os.path.isfile(self.target_actor_network_file):
            self.target_policy = keras.models.load_model(self.target_actor_network_file)
            print("Target Actor Model restored from checkpoint.")
        else:
            self.target_policy = policy_network(state_space_shape[0], action_space_shape[0])
            self.target_policy.set_weights(self.actor.get_weights())

        if os.path.isfile(self.critic_network_file):
            self.critic = keras.models.load_model(self.critic_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic = critic_network(state_space_shape[0], action_space_shape[0])
        if os.path.isfile(self.target_critic_network_file):
            self.target_critic = keras.models.load_model(self.target_critic_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_critic = critic_network(state_space_shape[0], action_space_shape[0])
            self.target_critic.set_weights(self.critic.get_weights())

    def validate(self):
        env = gym.make('LunarLanderContinuous-v2')
        done = False
        observation = env.reset()

        episodic_reward = 0

        while not done:
            #env.render()
            chosen_action = self.actor(np.expand_dims(observation, axis = 0), training=False)[0].numpy()
            next_observation, reward, done, _ = env.step(chosen_action)
            observation = next_observation
            episodic_reward += reward
        env.close()
        print(f'\t\t[Learner] Validation run total reward = {episodic_reward}')
        return episodic_reward

    def run(self):
        self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS) #initial target networks distribution
        self.weights_pipe.send([self.actor.get_weights(), self.critic.get_weights()])

        while self.buffer_ready_flag.value < 1:
            sleep(1)

        rewards = []
        training_runs = 1
        while self.cancellation_token.value == 0:
            self.cmd_pipe.send(CMD_GET_REPLAY_DATA)
            batches = self.replay_data_pipe.recv()
            
            priorities_updates = []
            for b in batches:
                actor_loss, critic_loss, td_errors = self.__train_actor_critic(b[0],b[1],b[2],b[3],b[4],b[5],b[6])
                priorities_updates.append((b[7], td_errors))
            
            self.cmd_pipe.send(CMD_UPDATE_PRIORITIES)
            self.priorities_pipe.send(priorities_updates)
            
            if self.training_active.value == 0:
                self.training_active.value = 1
            if training_runs % 5 == 0:
                self.__soft_update_models()
            if training_runs % 20 == 0:
                rewards.append(self.validate())
                if np.mean(rewards[-100:]) >= self.finish_criteria:
                    self.cancellation_token.value = 1
            if training_runs % 10 == 0:
                self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS)
                self.weights_pipe.send([self.actor.get_weights(), self.critic.get_weights()])
            if training_runs % self.checkpoint_step == 0:
                self.actor.save(self.actor_network_file)
                self.critic.save(self.critic_network_file)
                self.target_policy.save(self.target_actor_network_file)
                self.target_critic.save(self.target_critic_network_file)
                print(f'\t\t[Learner] Checkpoint saved on {training_runs} step')
            
            training_runs += 1
        print('\t\t[Learner] training complete.')

    @tf.function
    def __train_actor_critic(self, states, actions, next_states, rewards, gamma_powers, dones, is_weights):
        target_mu = self.target_policy(next_states, training=False)
        target_q = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * tf.reduce_max((1 - dones) * self.target_critic([next_states, target_mu], training=False), axis = 1)

        with tf.GradientTape() as tape:
            current_q = self.critic([states, actions], training=True)
            td_errors = target_q - tf.squeeze(current_q, axis=1)
            c_loss = tf.reduce_mean(is_weights * tf.math.pow(td_errors, 2), axis=0)
        gradients = tape.gradient(c_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            current_mu = self.actor(states, training=True)
            current_q = self.critic([states, current_mu], training=False)
            a_loss = tf.reduce_mean(-current_q)
        gradients = tape.gradient(a_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        return a_loss, c_loss, tf.math.abs(td_errors)

    def __soft_update_models(self):
        target_actor_weights = self.target_policy.get_weights()
        actor_weights = self.actor.get_weights()
        updated_actor_weights = []
        for aw,taw in zip(actor_weights,target_actor_weights):
            updated_actor_weights.append(self.tau * aw + (1.0 - self.tau) * taw)
        self.target_policy.set_weights(updated_actor_weights)

        target_critic_weights = self.target_critic.get_weights()
        critic_weights = self.critic.get_weights()
        updated_critic_weights = []
        for cw,tcw in zip(critic_weights,target_critic_weights):
            updated_critic_weights.append(self.tau * cw + (1.0 - self.tau) * tcw)
        self.target_critic.set_weights(updated_critic_weights)

def RunLearner(batch_size:int, gamma:float, actor_leraning_rate:float, critic_learning_rate:float,
               state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...],
                cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                cancelation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value):
    learner = Learner(batch_size, gamma, actor_leraning_rate, critic_learning_rate, 
                      state_space_shape, action_space_shape,
                      cmd_pipe, weights_pipe, replay_data_pipe, priorities_pipe, 
                      cancelation_token, training_active_flag, buffer_ready)
    learner.run()