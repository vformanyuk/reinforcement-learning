import gym
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from tensorflow import keras
from time import sleep
from typing import Tuple

from R2D2.DTOs import LearnerTransmitionBuffer
from R2D2.neural_networks import policy_network, critic_network

CMD_SET_NETWORK_WEIGHTS = 0
CMD_GET_REPLAY_DATA = 1
CMD_UPDATE_PRIORITIES = 2

class Learner(object):
    def __init__(self, batch_size:float, gamma:float, actor_learning_rate:float, critic_learning_rate:float, 
                 state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...], recurrent_layer_size:int, 
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                 cancellation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value,
                 *args, **kwargs):
        self.cancellation_token = cancellation_token
        self.training_active = training_active_flag
        self.buffer_ready_flag = buffer_ready

        self.logging_enabled = True

        # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.005
        self.gradient_step = 1
        self.finish_criteria = 200
        self.checkpoint_step = 100
        self.rnn_size = recurrent_layer_size
        self.stack_size = 4
        self.trajectory_n = 0.9

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
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.gaus_distr = tfp.distributions.Normal(0,1)

        self.alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)
        self.target_entropy = -2
        actor_recurrent_layer_size = 256

        self.actor_network_file = "r2d2-sac-learner-actor.h5"
        self.critic1_network_file = "r2d2-sac-learner-critic1.h5"
        self.target_critic1_network_file = "r2d2-sac-learner-target_critic1.h5"
        self.critic2_network_file = "r2d2-sac-learner-critic2.h5"
        self.target_critic2_network_file = "r2d2-sac-learner-target_critic2.h5"

        if os.path.isfile(self.actor_network_file):
            self.actor = keras.models.load_model(self.actor_network_file)
            print("Actor Model restored from checkpoint.")
        else:
            self.actor = policy_network(state_space_shape, action_space_shape[0], actor_recurrent_layer_size)

        if os.path.isfile(self.critic1_network_file):
            self.critic1 = keras.models.load_model(self.critic1_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic1 = critic_network(state_space_shape, action_space_shape[0], actor_recurrent_layer_size)
        if os.path.isfile(self.target_critic1_network_file):
            self.target_critic1 = keras.models.load_model(self.target_critic1_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_critic1 = critic_network(state_space_shape, action_space_shape[0], actor_recurrent_layer_size)
            self.target_critic1.set_weights(self.critic1.get_weights())

        if os.path.isfile(self.critic2_network_file):
            self.critic2 = keras.models.load_model(self.critic2_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic2 = critic_network(state_space_shape, action_space_shape[0], actor_recurrent_layer_size)
        if os.path.isfile(self.target_critic2_network_file):
            self.target_critic2 = keras.models.load_model(self.target_critic2_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_critic2 = critic_network(state_space_shape, action_space_shape[0], actor_recurrent_layer_size)
            self.target_critic2.set_weights(self.critic2.get_weights())

    def interpolation_step(self, env, s0, action, stack_size=4):
        result_states = []
        sN, r, d, _ = env.step(action)
        #interpolate between s0 and sN
        xp = [0, stack_size - 1]
        x = [i for i in range(stack_size) if i not in xp]
        interp_count = stack_size - 2
        result_states.append(s0)
        for _ in range(interp_count):
            result_states.append(np.zeros(shape=(len(s0)),dtype=np.float))
        result_states.append(sN)
        for i , y_boundary in enumerate(zip(s0, sN)):
            y_linear = np.interp(x, xp, y_boundary)
            for j, y in enumerate(y_linear):
                result_states[j+1][i] = y
        return result_states, r, d

    def log(self, msg):
        if self.logging_enabled:
            print(f'\t\t[Learner ({os.getpid()})]: {msg}')

    def validate(self):
        env = gym.make('LunarLanderContinuous-v2')
        done = False
        state0 = env.reset()
        observation = []
        for _ in range(self.stack_size):
            observation.append(state0)

        episodic_reward = 0
        episode_step = 0
        actor_hx = tf.zeros(shape=(1, self.rnn_size), dtype=tf.float32)

        while not done:
            mean, log_std_dev, actor_hx = self.actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
            throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0])
            eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1])

            next_observation, reward, done = self.interpolation_step(env, state0, [throttle_action, eng_ctrl_action], self.stack_size)
            state0 = next_observation[-1:][0]

            observation = next_observation
            episodic_reward += reward
            episode_step += 1
        env.close()
        self.log(f'Validation run: {episode_step} steps, total reward = {episodic_reward}')
        return episodic_reward

    def run(self):
        self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS) #initial target networks distribution
        self.weights_pipe.send([self.actor.get_weights(), self.critic1.get_weights(), self.critic2.get_weights(), self.alpha_log.numpy()])

        while self.buffer_ready_flag.value < 1:
            sleep(1)

        self.log("Training in progress")
        episode_rewards = []
        training_runs = 1
        while self.cancellation_token.value == 0:
            self.cmd_pipe.send(CMD_GET_REPLAY_DATA)
            trajectories:LearnerTransmitionBuffer = self.replay_data_pipe.recv()
            
            td_errors = dict()
            actor_losses = []
            critic_losses = []

            # actor_h and meta_idx are single tensors. Others are mini batches of values
            for actor_h, burn_in_states, states, actions, next_states, rewards, gamma_powers, dones, is_weights, meta_idx in trajectories:
                actor_training_hx = self.actor_burn_in(burn_in_states, actor_h, tf.convert_to_tensor(len(rewards), dtype=tf.int32))
                for _ in range(self.gradient_step):
                    critic1_loss, critic2_loss = self.train_critics(actor_training_hx, states, actions, next_states, rewards, gamma_powers, is_weights, dones)
                    critic_losses.append(critic1_loss)
                    critic_losses.append(critic2_loss)
                    actor_loss = self.train_actor(states, actor_training_hx)
                    actor_losses.append(actor_loss)
                td_errors[meta_idx] = self.get_trajectory_error(states, actions, next_states, rewards, gamma_powers, dones, actor_training_hx)
            self.log(f'Critic error {np.mean(critic_losses):.4f} Actor error {np.mean(actor_losses):.4f}')
            
            reversed_idxs = list(td_errors.keys())
            reversed_idxs.sort(reverse = True)

            self.cmd_pipe.send(CMD_UPDATE_PRIORITIES)
            self.priorities_pipe.send((reversed_idxs, list([td_errors[idx] for idx in reversed_idxs]))) 

            self.soft_update_models()
            
            if self.training_active.value == 0:
                self.training_active.value = 1
            if training_runs % 20 == 0:
                episode_rewards.append(self.validate())
                if np.mean(episode_rewards[-100:]) >= self.finish_criteria:
                    self.cancellation_token.value = 1
            if training_runs % 10 == 0:
                self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS)
                self.weights_pipe.send([self.actor.get_weights(), self.critic1.get_weights(), self.critic2.get_weights(), self.alpha_log])
            if training_runs % self.checkpoint_step == 0:
                self.actor.save(self.actor_network_file)
                self.critic1.save(self.critic1_network_file)
                self.critic2.save(self.critic2_network_file)
                self.target_critic1.save(self.target_critic1_network_file)
                self.target_critic2.save(self.target_critic2_network_file)
                self.log(f'Checkpoint saved on {training_runs} step')
            
            training_runs += 1
        self.log('training complete.')
    
    @tf.function(experimental_relax_shapes=True)
    def actor_burn_in(self, states, hx0, trajectory_length):
        hx = tf.expand_dims(hx0, axis=0)
        for s in states:
            _, __, hx = self.actor([tf.expand_dims(s, axis = 0), hx], training=False)
        return tf.tile(hx, [trajectory_length, 1])

    @tf.function(experimental_relax_shapes=True)
    def get_actions(self, mu, log_sigma):
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * self.gaus_distr.sample())

    @tf.function(experimental_relax_shapes=True)
    def get_log_probs(self, mu, sigma, actions):
        action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        log_probs = action_distributions.log_prob(mu + sigma*self.gaus_distr.sample()) - tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + 1e-6), axis=1)
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def actor_burn_in(self, states, hx0, trajectory_length):
        hx = hx0# hx = tf.expand_dims(hx0, axis=0)
        for s in states:
            _, __, hx = self.actor([tf.expand_dims(s, axis = 0), hx], training=False)
        return tf.tile(hx, [trajectory_length, 1])

    @tf.function(experimental_relax_shapes=True)
    def train_critics(self, actor_hx, states, actions, next_states, rewards, gamma_powers, is_weights, dones):
        mu, log_sigma, ___ = self.actor([next_states, actor_hx], training=False)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

        target_actions = self.get_actions(mu, log_sigma)
        target_actions_shape = tf.shape(target_actions)
        if len(target_actions_shape)  < 2:
            target_actions = tf.expand_dims(target_actions, axis=0)

        min_q = tf.math.minimum(self.target_critic1([next_states, target_actions], training=False), \
                                self.target_critic2([next_states, target_actions], training=False))
        min_q = tf.squeeze(min_q, axis=1)

        sigma = tf.math.exp(log_sigma)
        log_probs = self.get_log_probs(mu, sigma, target_actions)
        next_values = min_q - tf.math.exp(self.alpha_log) * log_probs

        target_q = rewards + tf.pow(self.gamma, gamma_powers) * (1 - dones) * next_values

        with tf.GradientTape() as tape:
            current_q = self.critic1([states, actions], training=True)
            c1_loss = 0.5 * tf.reduce_mean(is_weights * tf.pow(target_q - current_q, 2))
        gradients = tape.gradient(c1_loss, self.critic1.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic1.trainable_variables))

        with tf.GradientTape() as tape:
            current_q = self.critic2([states, actions], training=True)
            c2_loss = 0.5 * tf.reduce_mean(is_weights * tf.pow(target_q - current_q, 2))
        gradients = tape.gradient(c2_loss, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic2.trainable_variables))
        return c1_loss, c2_loss

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, states, hidden_rnn_states):
        alpha = tf.math.exp(self.alpha_log)
        with tf.GradientTape() as tape:
            mu, log_sigma, ___ = self.actor([states, hidden_rnn_states], training=True)
            mu = tf.squeeze(mu)
            log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

            target_actions = self.get_actions(mu, log_sigma)
            target_actions_shape = tf.shape(target_actions)
            if len(target_actions_shape)  < 2:
                target_actions = tf.expand_dims(target_actions, axis=0)
            
            target_q = tf.math.minimum(self.critic1([states, target_actions], training=False), \
                                    self.critic2([states, target_actions], training=False))
            target_q = tf.squeeze(target_q, axis=1)
            
            sigma = tf.math.exp(log_sigma)
            log_probs = self.get_log_probs(mu, sigma, target_actions)

            actor_loss = tf.reduce_mean(alpha * log_probs - target_q)
            
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean(self.alpha_log * tf.stop_gradient(log_probs + self.target_entropy))
            alpha_gradients = alpha_tape.gradient(alpha_loss, self.alpha_log)
            self.alpha_optimizer.apply_gradients([(alpha_gradients, self.alpha_log)])

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        return actor_loss

    @tf.function(experimental_relax_shapes=True)
    def get_trajectory_error(self, states, actions, next_states, rewards, gamma_powers, dones, hidden_rnn_states):
        mu, log_sigma, ___ = self.actor([next_states, hidden_rnn_states], training=False)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

        next_actions = self.get_actions(mu, log_sigma)
        next_actions_shape = tf.shape(next_actions)
        if len(next_actions_shape)  < 2:
            next_actions = tf.expand_dims(next_actions, axis=0)
        
        target_q = tf.math.minimum(self.critic1([next_states, next_actions], training=False), \
                                self.critic2([next_states, next_actions], training=False))
        target_q = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * tf.squeeze(target_q, axis=1)

        current_q = tf.math.minimum(self.critic1([states, actions], training=False), \
                                    self.critic2([states, actions], training=False))

        td_errors = target_q - tf.squeeze(current_q, axis=1)
        td_errors_shape = tf.shape(td_errors)
        if len(td_errors_shape) == 0:
            return td_errors

        return tf.reduce_max(td_errors) * self.trajectory_n + (1-self.trajectory_n)*tf.reduce_mean(td_errors)

    def soft_update_models(self):
        target_critic1_weights = self.target_critic1.get_weights()
        critic1_weights = self.critic1.get_weights()
        updated_target_critic_weights = []
        for c1w,tc1w in zip(critic1_weights,target_critic1_weights):
            updated_target_critic_weights.append(self.tau * c1w + (1.0 - self.tau) * tc1w)
        self.target_critic1.set_weights(updated_target_critic_weights)

        target_critic2_weights = self.target_critic2.get_weights()
        critic2_weights = self.critic2.get_weights()
        updated_target_critic_weights = []
        for c2w,tc2w in zip(critic2_weights,target_critic2_weights):
            updated_target_critic_weights.append(self.tau * c2w + (1.0 - self.tau) * tc2w)
        self.target_critic2.set_weights(updated_target_critic_weights)

def RunLearner(batch_size:int, gamma:float, actor_leraning_rate:float, critic_learning_rate:float,
               state_space_shape:Tuple[float,...], action_space_shape:Tuple[float,...], recurrent_layer_size:int,
                cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, priorities_pipe:mp.Pipe,
                cancelation_token:mp.Value, training_active_flag:mp.Value, buffer_ready:mp.Value):
    learner = Learner(batch_size, gamma, actor_leraning_rate, critic_learning_rate, 
                      state_space_shape, action_space_shape, recurrent_layer_size,
                      cmd_pipe, weights_pipe, replay_data_pipe, priorities_pipe, 
                      cancelation_token, training_active_flag, buffer_ready)
    learner.run()
