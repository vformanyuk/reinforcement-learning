import gym
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from tensorflow import keras
from time import sleep
from typing import Tuple

from APEX.neural_networks import sac_policy_network, sac_critic_network, sac_value_network

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
        self.tau = 0.005
        self.gradient_step = 1
        self.finish_criteria = 200
        self.checkpoint_step = 100
        self.action_space_shape = action_space_shape[0]

        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_data_pipe = replay_data_pipe
        self.priorities_pipe = priorities_pipe

        self.log_std_min=-20
        self.log_std_max=2

        RND_SEED = 0x12345
        tf.random.set_seed(RND_SEED)
        np.random.random(RND_SEED)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.gaus_distr = tfp.distributions.Normal(0,1)

        self.alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)
        self.target_entropy = -2

        self.actor_network_file = "apex-sac-learner-actor.h5"
        self.critic1_network_file = "apex-sac-learner-critic1.h5"
        self.critic2_network_file = "apex-sac-learner-critic2.h5"
        self.value_network_file = "apex-sac-value.h5"
        self.value_target_network_file = "apex-sac-value_target.h5"

        if os.path.isfile(self.actor_network_file):
            self.actor = keras.models.load_model(self.actor_network_file)
            print("Actor Model restored from checkpoint.")
        else:
            self.actor = sac_policy_network(state_space_shape[0], self.action_space_shape)

        if os.path.isfile(self.critic1_network_file):
            self.critic1 = keras.models.load_model(self.critic1_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic1 = sac_critic_network(state_space_shape[0], self.action_space_shape)
        if os.path.isfile(self.critic2_network_file):
            self.critic2 = keras.models.load_model(self.critic2_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic2 = sac_critic_network(state_space_shape[0], self.action_space_shape)

        if os.path.isfile(self.value_network_file):
            self.value_net = keras.models.load_model(self.value_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.value_net = sac_value_network(state_space_shape[0])
        if os.path.isfile(self.value_target_network_file):
            self.target_value_net = keras.models.load_model(self.value_target_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_value_net = sac_value_network(state_space_shape[0])
            self.target_value_net.set_weights(self.value_net.get_weights())

    def validate(self):
        env = gym.make('LunarLanderContinuous-v2')
        done = False
        observation = env.reset()

        episodic_reward = 0

        while not done:
            #env.render()
            mean, log_std_dev = self.actor(np.expand_dims(observation, axis = 0), training=False)
            throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0], self.gaus_distr.sample())
            eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1], self.gaus_distr.sample())

            next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])
            observation = next_observation
            episodic_reward += reward
        env.close()
        print(f'\t\t[Learner] Validation run total reward = {episodic_reward}')
        return episodic_reward

    def run(self):
        self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS) #initial target networks distribution
        self.weights_pipe.send([self.actor.get_weights(), self.value_net.get_weights()])

        while self.buffer_ready_flag.value < 1:
            sleep(1)

        rewards = []
        training_runs = 1
        while self.cancellation_token.value == 0:
            self.cmd_pipe.send(CMD_GET_REPLAY_DATA)
            batches = self.replay_data_pipe.recv()
            
            priorities_updates = []
            for b in batches:
                for _ in range(self.gradient_step):
                    critic1_loss, critic2_loss, td_errors = self.__train_critics(b[0],b[1],b[2],b[3],b[4],b[5],b[6])
                    actor_loss, value_loss = self.__train_actor_and_value(b[0])
                    priorities_updates.append((b[7], td_errors))
            
            self.cmd_pipe.send(CMD_UPDATE_PRIORITIES)
            self.priorities_pipe.send(priorities_updates)

            self.__soft_update_models()
            
            if self.training_active.value == 0:
                self.training_active.value = 1
            if training_runs % 20 == 0:
                rewards.append(self.validate())
                if np.mean(rewards[-100:]) >= self.finish_criteria:
                    self.cancellation_token.value = 1
            if training_runs % 10 == 0:
                self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS)
                self.weights_pipe.send([self.actor.get_weights(), self.value_net.get_weights()])
            if training_runs % self.checkpoint_step == 0:
                self.actor.save(self.actor_network_file)
                self.critic1.save(self.critic1_network_file)
                self.critic2.save(self.critic2_network_file)
                self.value_net.save(self.value_network_file)
                self.target_value_net.save(self.value_target_network_file)
                print(f'\t\t[Learner] Checkpoint saved on {training_runs} step')
            
            training_runs += 1
        print('\t\t[Learner] training complete.')

    @tf.function
    def get_actions(self, mu, log_sigma, noise):
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

    @tf.function
    def get_log_probs(self, mu, sigma, actions, noise):
        action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        # appendix C of the SAC paper discribe applyed boundings which is log(1-tanh(u)^2)
        log_probs = action_distributions.log_prob(mu + sigma*noise) - tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + 1e-6), axis=1)
        return log_probs

    @tf.function
    def __train_critics(self, states, actions, next_states, rewards, gamma_powers, dones, is_weights):
        next_values = tf.squeeze(self.target_value_net(next_states, training=False), axis = 1)
        target_v = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * next_values

        with tf.GradientTape() as tape:
            current_q1 = self.critic1([states, actions], training=True)
            td_errors = target_v - tf.squeeze(current_q1, axis=1)
            c1_loss = tf.reduce_mean(is_weights * tf.math.pow(td_errors, 2), axis=0)
        gradients = tape.gradient(c1_loss, self.critic1.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic1.trainable_variables))

        with tf.GradientTape() as tape:
            current_q2 = self.critic2([states, actions], training=True)
            td_errors = target_v - tf.squeeze(current_q2, axis=1)
            c2_loss = tf.reduce_mean(is_weights * tf.math.pow(td_errors, 2), axis=0)
        gradients = tape.gradient(c2_loss, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic2.trainable_variables))

        td_error = target_v - tf.squeeze(self.value_net(states, training=False), axis=1)

        return c1_loss, c2_loss, tf.abs(td_error)

    @tf.function
    def __train_actor_and_value(self, states):
        alpha = tf.math.exp(self.alpha_log)
        noise = self.gaus_distr.sample(sample_shape=(self.batch_size, self.action_space_shape))
        with tf.GradientTape() as tape:
            mu, log_sigma = self.actor(states, training=True)
            mu = tf.squeeze(mu)
            log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

            target_actions = self.get_actions(mu, log_sigma, noise)
            log_probs = self.get_log_probs(mu, tf.math.exp(log_sigma), target_actions, noise)
        
            target_q = tf.math.minimum(self.critic1([states, target_actions], training=False), \
                                       self.critic2([states, target_actions], training=False))
            target_q = tf.squeeze(target_q, axis=1)
        
            actor_loss = tf.reduce_mean(alpha * log_probs - target_q)
        
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean(self.alpha_log * tf.stop_gradient(log_probs + self.target_entropy))
            alpha_gradients = alpha_tape.gradient(alpha_loss, self.alpha_log)
            self.alpha_optimizer.apply_gradients([(alpha_gradients, self.alpha_log)])
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        target_v = target_q - tf.exp(self.alpha_log) * log_probs
        with tf.GradientTape() as value_tape:
            current_v = self.value_net(states, training=True)
            value_loss = self.mse_loss(current_v, target_v)
        value_gradient = value_tape.gradient(value_loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradient, self.value_net.trainable_variables))
        return actor_loss, value_loss

    def __soft_update_models(self):
        target_value_weights = self.target_value_net.get_weights()
        value_weights = self.value_net.get_weights()
        updated_weights = []
        for c2w,tc2w in zip(value_weights, target_value_weights):
            updated_weights.append(self.tau * c2w + (1.0 - self.tau) * tc2w)
        self.target_value_net.set_weights(updated_weights)

def RunLearner(batch_size:int, gamma:float, actor_leraning_rate:float, critic_learning_rate:float,
               state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...],
                cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                cancelation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value):
    learner = Learner(batch_size, gamma, actor_leraning_rate, critic_learning_rate, 
                      state_space_shape, action_space_shape,
                      cmd_pipe, weights_pipe, replay_data_pipe, priorities_pipe, 
                      cancelation_token, training_active_flag, buffer_ready)
    learner.run()
