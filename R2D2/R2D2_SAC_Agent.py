from typing import List
import gym
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from R2D2.DTOs import AgentTransmitionBuffer
from R2D2.neural_networks import policy_network, critic_network, value_network
from R2D2.R2D2_AgentBuffer import R2D2_AgentBuffer

class Actor(object):
    def __init__(self, id:int, gamma:float,
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value,
                 *args, **kwargs):
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([], 'GPU') # run actors on CPU

        self.debug_mode = False
        self.id = id
        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_pipe = replay_pipe
        self.cancelation_token = cancelation_token
        self.training_active = training_active_flag
        self.exchange_steps = 100 + np.random.randint(low=10, high=90, size=1)[0]
        self.gamma = gamma
        self.q_rescaling_epsilone = tf.constant(1e-6, dtype=tf.float32)
        self.gaus_distr = tfp.distributions.Normal(0,1)
        self.stack_size = 4
        self.N = 5
        self.action_bounds_epsilon = 1e-6
        self.log_std_min = -20
        self.log_std_max = 2
        self.trajectory_n = 0.9
        self.trajectory_length = 40
        self.burn_in_length = 10
        self.actor_recurrent_layer_size = 512
        self.pid = os.getpid()

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor ({self.pid}) {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.actor.set_weights(weights[0])
            self.critic_1.set_weights(weights[1])
            self.critic_2.set_weights(weights[2])
            self.value_network.set_weights(weights[3])
            self.log(f'Target actor and target critic weights refreshed.')
        except EOFError:
            print("[get_target_weights] Connection closed.")
        except OSError:
            print("[get_target_weights] Connection closed.")

    def send_replay_data(self, data:AgentTransmitionBuffer):
        try:
            self.cmd_pipe.send([1, self.id])
            #self.log(f'Replay data command sent.')
            self.replay_pipe.send(data)
            self.log(f'Replay data sent.')
        except EOFError:
            print("[send_replay_data] Connection closed.")
        except OSError:
            print("[send_replay_data] Connection closed.")

    def prepare_and_send_replay_data(self, exp_buffer:R2D2_AgentBuffer, trajectories:List[int]):
        transmittion_buffer = AgentTransmitionBuffer()
        for burn_in_hidden, burn_in_states, burn_in_actions, states, actions, next_states, rewards, gamma_powers, dones, stored_hidden_states in exp_buffer.get_data(trajectories):
            trajectory_length = tf.convert_to_tensor(len(rewards), dtype=tf.int32)
            if len(burn_in_states) > 0:
                ch1, ch2 = self.networks_rollout(burn_in_states, burn_in_actions, trajectory_length)
            else:
                ch1 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
                ch2 = tf.zeros(shape=(trajectory_length, self.actor_recurrent_layer_size), dtype=tf.float32)
            td_errors = self.get_trajectory_error(states, actions, next_states, rewards, gamma_powers, dones, ch1, ch2)
            transmittion_buffer.append(burn_in_hidden, burn_in_states, burn_in_actions, 
                                        states, actions, next_states, rewards, gamma_powers, dones, 
                                        stored_hidden_states, td_errors)
        if len(transmittion_buffer) > 0:
            self.send_replay_data(transmittion_buffer)

    @tf.function(experimental_relax_shapes=True)
    def networks_rollout(self, states, actions, trajectory_length):
        chx1 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        chx2 = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)
        for i in range(len(states)):
            _, chx1 = self.critic_1([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), chx1], training=False)
            _, chx2 = self.critic_2([tf.expand_dims(states[i], axis = 0), tf.expand_dims(actions[i], axis = 0), chx2], training=False)
        return tf.tile(chx1, [trajectory_length, 1]), tf.tile(chx2, [trajectory_length, 1])

    @tf.function(experimental_relax_shapes=True)
    def get_actions(self, mu, log_sigma):
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * self.gaus_distr.sample())

    @tf.function(experimental_relax_shapes=True)
    def get_log_probs(self, mu, sigma, actions):
        action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        log_probs = action_distributions.log_prob(mu + sigma * self.gaus_distr.sample()) - \
                    tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + self.action_bounds_epsilon), axis=1)
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def invertible_function_rescaling(self, x):
        return tf.sign(x)*(tf.sqrt(tf.abs(x) + 1) - 1) + self.q_rescaling_epsilone * x

    @tf.function(experimental_relax_shapes=True)
    def get_trajectory_error(self, states, actions, next_states, rewards, gamma_powers, dones, ch1, ch2):
        q1, __ = self.critic_1([states, actions, ch1], training=False)
        q2, __ = self.critic_2([states, actions, ch2], training=False)
        current_q = tf.math.minimum(q1, q2)
        
        target_v_estimation = self.value_network(next_states, training = False)

        inverse_v_rescaling = 1 / self.invertible_function_rescaling(tf.squeeze(target_v_estimation, axis=1))
        target_v = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * inverse_v_rescaling
        target_v = self.invertible_function_rescaling(target_v)

        td_errors = tf.abs(target_v - tf.squeeze(current_q, axis=1))

        # td_errors_shape = tf.shape(td_errors)
        # if len(td_errors_shape) == 0:
        #     return td_errors

        priority = tf.reduce_max(td_errors) * self.trajectory_n + (1 - self.trajectory_n) * tf.reduce_mean(td_errors)
        return priority

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

    def _on_trajectory_ready(self, buffer:R2D2_AgentBuffer, idx:int):
        self.prepare_and_send_replay_data(buffer, [idx])

    def run(self):
        env = gym.make('LunarLanderContinuous-v2')
        state_space_shape = (self.stack_size, env.observation_space.shape[0])
        outputs_count = env.action_space.shape[0]

        self.actor = policy_network(state_space_shape, outputs_count, self.actor_recurrent_layer_size)
        
        self.critic_1 = critic_network(state_space_shape, outputs_count, self.actor_recurrent_layer_size)
        self.critic_2 = critic_network(state_space_shape, outputs_count, self.actor_recurrent_layer_size)

        self.value_network = value_network(state_space_shape)

        self.get_target_weights()

        exp_buffer = R2D2_AgentBuffer(distributed_mode=True, buffer_size=1001, N=self.N, gamma=self.gamma, 
                                    state_shape=(self.stack_size, env.observation_space.shape[0]),
                                    action_shape=env.action_space.shape, 
                                    trajectory_ready_callback=self._on_trajectory_ready, 
                                    trajectory_size=self.trajectory_length, burn_in_length=self.burn_in_length)
        rewards_history = []
        
        global_step = 0
        for i in range(50000):
            if self.cancelation_token.value != 0:
                break
            
            done = False
            state0 = env.reset()
            observation = []
            for _ in range(self.stack_size):
                observation.append(state0)

            exp_buffer.reset()
            actor_hx = tf.zeros(shape=(1, self.actor_recurrent_layer_size), dtype=tf.float32)

            episodic_reward = 0
            epoch_steps = 0

            while not done and self.cancelation_token.value == 0:
                mean, log_std_dev, actor_hx = self.actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
                throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0])
                eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1])

                next_observation, reward, done = self.interpolation_step(env, state0, [throttle_action, eng_ctrl_action])
                state0 = next_observation[-1:][0]

                exp_buffer.store(actor_hx, observation, [throttle_action, eng_ctrl_action], reward, float(done))

                if done:
                    self.prepare_and_send_replay_data(exp_buffer, exp_buffer.get_remaining_trajectories())

                if global_step % self.exchange_steps == 0 and self.training_active.value > 0: # update target networks every 'exchange_steps'
                    self.get_target_weights()
                    
                observation = next_observation
                global_step+=1
                epoch_steps+=1
                episodic_reward += reward

            rewards_history.append(episodic_reward)
            last_mean = np.mean(rewards_history[-100:])
            self.log(f'[{i} ({epoch_steps})] Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
            if last_mean > 200:
                self.actor.save(f'lunar_lander_R2D2_{self.id}.h5')
                break
        env.close()
        print(f'Agent [{self.id}] done training.')

def RunActor(id:int, gamma:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = Actor(id, gamma, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()
