import gym
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from tensorflow import keras
from time import sleep
from typing import Tuple

# from R2D2.LearningRateDecayScheduler import LearningRateDecay
from R2D2.DTOs import LearnerTransmitionBuffer
from R2D2.neural_networks import policy_network, critic_network, value_network

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
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.005
        self.gradient_step = 2
        self.finish_criteria = 200
        self.checkpoint_step = 10 * self.batch_size
        self.validation_step = 3 * self.batch_size
        self.networks_transmite_step = 5 * self.batch_size
        self.rnn_size = recurrent_layer_size
        self.stack_size = 4
        self.trajectory_n = 0.9
        self.q_rescaling_epsilone = tf.constant(1e-6, dtype=tf.float32)
        self.action_space_shape = action_space_shape
        self.state_space_shape = state_space_shape

        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_data_pipe = replay_data_pipe
        self.priorities_pipe = priorities_pipe

        self.log_std_min=-20
        self.log_std_max=2

        RND_SEED = 0x12345
        tf.random.set_seed(RND_SEED)
        np.random.random(RND_SEED)

        # self.actor_lr_scheduler = LearningRateDecay(actor_learning_rate)
        # self.critic_lr_scheduler = LearningRateDecay(critic_learning_rate)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.gaus_distr = tfp.distributions.Normal(0,1)

        self.alpha_log = tf.Variable(0.5, dtype = tf.float32, trainable=True)
        self.target_entropy = -2
        self.actor_recurrent_layer_size = 512

        self.actor_network_file = "r2d2-sac-learner-actor.h5"
        self.critic1_network_file = "r2d2-sac-learner-critic1.h5"
        self.target_critic1_network_file = "r2d2-sac-learner-target_critic1.h5"
        self.critic2_network_file = "r2d2-sac-learner-critic2.h5"
        self.target_critic2_network_file = "r2d2-sac-learner-target_critic2.h5"
        self.value_network_file = "r2d2-sac-value.h5"

        if os.path.isfile(self.actor_network_file):
            self.actor = keras.models.load_model(self.actor_network_file)
            print("Actor Model restored from checkpoint.")
        else:
            self.actor = policy_network(state_space_shape, action_space_shape[0], self.actor_recurrent_layer_size)

        if os.path.isfile(self.critic1_network_file):
            self.critic1 = keras.models.load_model(self.critic1_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic1 = critic_network(state_space_shape, action_space_shape[0], self.actor_recurrent_layer_size)
        if os.path.isfile(self.target_critic1_network_file):
            self.target_critic1 = keras.models.load_model(self.target_critic1_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_critic1 = critic_network(state_space_shape, action_space_shape[0], self.actor_recurrent_layer_size)
            self.target_critic1.set_weights(self.critic1.get_weights())

        if os.path.isfile(self.critic2_network_file):
            self.critic2 = keras.models.load_model(self.critic2_network_file)
            print("Critic Model restored from checkpoint.")
        else:
            self.critic2 = critic_network(state_space_shape, action_space_shape[0], self.actor_recurrent_layer_size)
        if os.path.isfile(self.target_critic2_network_file):
            self.target_critic2 = keras.models.load_model(self.target_critic2_network_file)
            print("Target Critic Model restored from checkpoint.")
        else:
            self.target_critic2 = critic_network(state_space_shape, action_space_shape[0], self.actor_recurrent_layer_size)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if os.path.isfile(self.value_network_file):
            self.value_net = keras.models.load_model(self.value_network_file)
            print("Value Model restored from checkpoint.")
        else:
            self.value_net = value_network(state_space_shape)

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
        sqrt_two_p_e = np.sqrt(2* np.pi * np.e)
        throttle_e = []
        ctrl_e = []

        validation_actor = policy_network(self.state_space_shape, self.action_space_shape[0], self.actor_recurrent_layer_size)
        validation_actor.set_weights(self.actor.get_weights())

        while not done:
            mean, log_sigma, actor_hx = validation_actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
            throttle_action = self.get_actions(mean[0][0], log_sigma[0][0], self.gaus_distr.sample())
            eng_ctrl_action = self.get_actions(mean[0][1], log_sigma[0][1], self.gaus_distr.sample())

            throttle_e.append(np.log(sqrt_two_p_e * np.exp(log_sigma[0][0])))
            ctrl_e.append(np.log(sqrt_two_p_e * np.exp(log_sigma[0][1])))

            next_observation, reward, done = self.interpolation_step(env, state0, [throttle_action, eng_ctrl_action], self.stack_size)
            state0 = next_observation[-1:][0]

            observation = next_observation
            episodic_reward += reward
            episode_step += 1
        env.close()
        if episodic_reward > 200:
            validation_actor.save("lunar_lander_r2d2_sac.h5")
        self.log(f'Validation run: {episode_step} steps, total reward = {episodic_reward}, throttle_e = {np.mean(throttle_e):.4f}, ctrl_e = {np.mean(ctrl_e):.4f}')
        return episodic_reward

    def run(self):
        self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS) #initial target networks distribution
        self.weights_pipe.send([self.actor.get_weights(), self.critic1.get_weights(), self.critic2.get_weights(), self.value_net.get_weights(), self.alpha_log.numpy()])

        while self.buffer_ready_flag.value < 1:
            sleep(1)

        self.log("Training in progress")
        episode_rewards = []
        training_runs = 0
        while self.cancellation_token.value == 0:
            self.cmd_pipe.send(CMD_GET_REPLAY_DATA)
            trajectories:LearnerTransmitionBuffer = self.replay_data_pipe.recv()
            
            td_errors = dict()
            actor_losses = []
            critic_losses = []
            value_losses = []

            # actor_h and meta_idx are single tensors. Others are mini batches of values
            for actor_h, burn_in_states, burn_in_actions, states, actions, next_states, rewards, gamma_powers, dones, stored_actor_states, is_weights, meta_idx in trajectories:
                trajectory_length = tf.convert_to_tensor(len(rewards), dtype=tf.int32)
                if len(burn_in_states) > 0:
                    ch1, ch2, target_ch1, target_ch2 = self.networks_rollout(burn_in_states, burn_in_actions, actor_h, trajectory_length)
                else:
                    ch1 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
                    ch2 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
                    target_ch1 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
                    target_ch2 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
                noise = self.gaus_distr.sample(sample_shape=(len(rewards), 2))
                for _ in range(self.gradient_step):
                    actor_loss, value_loss, next_hidden_states = self.train_actor_and_value(states, noise, stored_actor_states, ch1, ch2)
                    value_losses.append(value_loss)
                    actor_losses.append(actor_loss)

                    critic1_loss, critic2_loss, priority = self.train_critics(states, actions, next_states, rewards, gamma_powers, is_weights, dones, 
                                                                                noise, next_hidden_states, ch1, ch2, target_ch1, target_ch2)
                    critic_losses.append(critic1_loss)
                    critic_losses.append(critic2_loss)
                    td_errors[meta_idx] = priority
                training_runs += 1
                self.soft_update_models()
            self.log(f'Critic error {np.mean(critic_losses):.4f} Value error {np.mean(value_losses):.4f} Actor error {np.mean(actor_losses):.4f}')
            
            reversed_idxs = list(td_errors.keys())
            reversed_idxs.sort(reverse = True)

            self.cmd_pipe.send(CMD_UPDATE_PRIORITIES)
            self.priorities_pipe.send((reversed_idxs, list([td_errors[idx] for idx in reversed_idxs]))) 

            if self.training_active.value == 0:
                self.training_active.value = 1
            if training_runs % self.validation_step == 0:
                episode_rewards.append(self.validate())
                if np.mean(episode_rewards[-100:]) >= self.finish_criteria:
                    self.cancellation_token.value = 1
            if training_runs % self.networks_transmite_step == 0:
                self.cmd_pipe.send(CMD_SET_NETWORK_WEIGHTS)
                self.weights_pipe.send([self.actor.get_weights(), self.critic1.get_weights(), self.critic2.get_weights(), self.value_net.get_weights(), self.alpha_log])
            if training_runs % self.checkpoint_step == 0:
                self.actor.save(self.actor_network_file)
                self.critic1.save(self.critic1_network_file)
                self.critic2.save(self.critic2_network_file)
                self.target_critic1.save(self.target_critic1_network_file)
                self.target_critic2.save(self.target_critic2_network_file)
                self.value_net.save(self.value_network_file)
                self.log(f'Checkpoint saved on {training_runs} step')
            
        self.log('training complete.')
    
    @tf.function(experimental_relax_shapes=True)
    def networks_rollout(self, states, actions, hx0, trajectory_length):
        ahx = hx0
        chx1 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        chx2 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        target_chx1 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        target_chx2 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        for i in range(len(states)):
            _, __, ahx = self.actor([tf.expand_dims(states[i], axis = 0), ahx], training=False)
            _, chx1 = self.critic1([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), chx1], training=False)
            _, chx2 = self.critic2([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), chx2], training=False)
            _, target_chx1 = self.target_critic1([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), target_chx1], training=False)
            _, target_chx2 = self.target_critic2([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), target_chx2], training=False)
        return  tf.tile(chx1, [trajectory_length, 1]), \
                tf.tile(chx2, [trajectory_length, 1]), \
                tf.tile(target_chx1, [trajectory_length, 1]), \
                tf.tile(target_chx2, [trajectory_length, 1])

    @tf.function(experimental_relax_shapes=True)
    def get_actions(self, mu, log_sigma, noise):
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

    @tf.function(experimental_relax_shapes=True)
    def get_log_probs(self, mu, sigma, actions, noise):
        action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        log_probs = action_distributions.log_prob(mu + sigma * noise) - tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + 1e-6), axis=1)
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def train_critics(self, states, actions, next_states, rewards, gamma_powers, is_weights, dones, noise,
                        actor_hs, critic1_hs, critic2_hs, target_critic1_hs, target_critic2_hs):
        mu, log_sigma, ___ = self.actor([next_states, actor_hs], training=False)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

        target_actions = self.get_actions(mu, log_sigma, noise)
        target_actions_shape = tf.shape(target_actions)
        if len(target_actions_shape)  < 2:
            target_actions = tf.expand_dims(target_actions, axis=0)

        min_q = tf.math.minimum(self.target_critic1([next_states, target_actions, target_critic1_hs], training=False)[0], \
                                self.target_critic2([next_states, target_actions, target_critic2_hs], training=False)[0])
        min_q = tf.squeeze(min_q, axis=1)

        sigma = tf.math.exp(log_sigma)
        log_probs = self.get_log_probs(mu, sigma, target_actions, noise)
        next_values = min_q - tf.math.exp(self.alpha_log) * log_probs

        target_q = rewards + tf.pow(self.gamma, gamma_powers + 1) * (1 - dones) * next_values

        with tf.GradientTape() as tape:
            current_q1, _ = self.critic1([states, actions, critic1_hs], training=True)
            c1_loss = tf.reduce_mean(is_weights * tf.pow(target_q - current_q1, 2))
        gradients = tape.gradient(c1_loss, self.critic1.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic1.trainable_variables))

        with tf.GradientTape() as tape:
            current_q2, _ = self.critic2([states, actions, critic2_hs], training=True)
            c2_loss = tf.reduce_mean(is_weights * tf.pow(target_q - current_q2, 2))
        gradients = tape.gradient(c2_loss, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic2.trainable_variables))\

        target_v_estimation = self.value_net(next_states, training = False)

        inverse_v_rescaling = 1 / self.invertible_function_rescaling(tf.squeeze(target_v_estimation, axis=1))
        target_v = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * inverse_v_rescaling
        target_v = self.invertible_function_rescaling(target_v)

        td_errors = tf.abs(target_v - tf.squeeze(tf.minimum(current_q1, current_q2), axis=1))
        priority = tf.reduce_max(td_errors) * self.trajectory_n + (1 - self.trajectory_n) * tf.reduce_mean(td_errors)
        
        return c1_loss, c2_loss, priority

    @tf.function(experimental_relax_shapes=True)
    def train_actor_and_value(self, states, noise, actor_hs, critic1_hs, critic2_hs):
        alpha = tf.math.exp(self.alpha_log)
        with tf.GradientTape() as tape:
            mu, log_sigma, next_hidden_states = self.actor([states, actor_hs], training=True)
            mu = tf.squeeze(mu)
            log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

            target_actions = self.get_actions(mu, log_sigma, noise)
            target_actions_shape = tf.shape(target_actions)
            if len(target_actions_shape)  < 2:
                target_actions = tf.expand_dims(target_actions, axis=0)
            
            target_q = tf.math.minimum(self.critic1([states, target_actions, critic1_hs], training=False)[0], \
                                        self.critic2([states, target_actions, critic2_hs], training=False)[0])
            target_q = tf.squeeze(target_q, axis=1)
            
            sigma = tf.math.exp(log_sigma)
            log_probs = self.get_log_probs(mu, sigma, target_actions, noise)

            actor_loss = tf.reduce_mean(alpha * log_probs - target_q)
            
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean(self.alpha_log * tf.stop_gradient(log_probs + self.target_entropy))
            alpha_gradients = alpha_tape.gradient(alpha_loss, self.alpha_log)
            self.alpha_optimizer.apply_gradients([(alpha_gradients, self.alpha_log)])

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        with tf.GradientTape() as value_tape:
            target_v = target_q - log_probs
            current_v = self.value_net(states, training=True)
            value_loss = self.mse_loss(current_v, target_v)
        value_gradient = value_tape.gradient(value_loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradient, self.value_net.trainable_variables))

        return actor_loss, value_loss, next_hidden_states

    @tf.function(experimental_relax_shapes=True)
    def invertible_function_rescaling(self, x):
        return tf.sign(x)*(tf.sqrt(tf.abs(x) + 1) - 1) + self.q_rescaling_epsilone * x

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
