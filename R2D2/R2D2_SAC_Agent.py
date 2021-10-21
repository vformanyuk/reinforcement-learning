import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from R2D2.DTOs import AgentTransmitionBuffer
from R2D2.neural_networks import policy_network, critic_network
from R2D2.R2D2_AgentBuffer import R2D2_AgentBuffer

class Actor(object):
    def __init__(self, id:int, gamma:float,
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value,
                 *args, **kwargs):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.debug_mode = False
        self.id = id
        self.cmd_pipe = cmd_pipe
        self.weights_pipe = weights_pipe
        self.replay_pipe = replay_pipe
        self.cancelation_token = cancelation_token
        self.training_active = training_active_flag
        self.exchange_steps = 100 + np.random.randint(low=10, high=90, size=1)[0]
        self.data_send_steps = 50
        self.gamma = gamma
        self.gaus_distr = tfp.distributions.Normal(0,1)
        self.stack_size = 4
        self.N = 5
        self.action_bounds_epsilon = 1e-6
        self.log_std_min = -20
        self.log_std_max = 2
        self.trajectory_n = 0.9
        self.trajectory_length = 40
        self.burn_in_length = 10

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.actor.set_weights(weights[0])
            self.critic_1.set_weights(weights[1])
            self.critic_2.set_weights(weights[2])
            self.log(f'Target actor and target critic weights refreshed.')
        except EOFError:
            print("[get_target_weights] Connection closed.")
        except OSError:
            print("[get_target_weights] Connection closed.")

    def send_replay_data(self, data:AgentTransmitionBuffer):
        try:
            self.cmd_pipe.send([1, self.id])
            self.log(f'Replay data command sent.')
            self.replay_pipe.send(data)
            self.log(f'Replay data sent.')
        except EOFError:
            print("[send_replay_data] Connection closed.")
        except OSError:
            print("[send_replay_data] Connection closed.")

    def prepare_and_send_replay_data(self, exp_buffer:R2D2_AgentBuffer, batch_length:int):
        transmittion_buffer = AgentTransmitionBuffer()
        for actor_h, burn_in_states, states, actions, next_states, rewards, gamma_powers, dones in exp_buffer.get_tail(batch_length):
            td_errors = self.get_trajectory_error(states, actions, next_states, rewards, gamma_powers, dones, actor_h)
            transmittion_buffer.append(actor_h, burn_in_states, states, actions, next_states, rewards, gamma_powers, dones, td_errors)
        self.send_replay_data(transmittion_buffer)

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
    def get_trajectory_error(self, states, actions, next_states, rewards, gamma_powers, dones, hidden_rnn_states):
        mu, log_sigma, ___ = self.actor([next_states, hidden_rnn_states], training=False)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

        next_actions = self.get_actions(mu, log_sigma)
        next_actions_shape = tf.shape(next_actions)
        if len(next_actions_shape)  < 2:
            next_actions = tf.expand_dims(next_actions, axis=0)
        
        target_q = tf.math.minimum(self.critic_1([next_states, next_actions], training=False), \
                                   self.critic_2([next_states, next_actions], training=False))
        target_q = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * tf.squeeze(target_q, axis=1)

        current_q = tf.math.minimum(self.critic_1([states, actions], training=False), \
                                    self.critic_2([states, actions], training=False))

        td_errors = target_q - tf.squeeze(current_q, axis=1)

        td_errors_shape = tf.shape(td_errors)
        if len(td_errors_shape) == 0:
            return td_errors

        return tf.reduce_max(td_errors) * self.trajectory_n + (1 - self.trajectory_n) * tf.reduce_mean(td_errors)

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

    def run(self):
        env = gym.make('LunarLanderContinuous-v2')
        state_space_shape = (self.stack_size, env.observation_space.shape[0])
        outputs_count = env.action_space.shape[0]
        actor_recurrent_layer_size = 256

        self.actor = policy_network(state_space_shape, outputs_count, actor_recurrent_layer_size)
        
        self.critic_1 = critic_network(state_space_shape, outputs_count, actor_recurrent_layer_size)
        self.critic_2 = critic_network(state_space_shape, outputs_count, actor_recurrent_layer_size)

        self.get_target_weights()

        exp_buffer = R2D2_AgentBuffer(distributed_mode=True, buffer_size=1001, N=self.N, gamma=self.gamma, 
                                    state_shape=(self.stack_size, env.observation_space.shape[0]),
                                    action_shape=env.action_space.shape, 
                                    hidden_state_shape=(actor_recurrent_layer_size,), 
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

            episodic_reward = 0
            epoch_steps = 0

            while not done and self.cancelation_token.value == 0:
                mean, log_std_dev, actor_hx = self.actor([np.expand_dims(observation, axis = 0), actor_hx], training=False)
                throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0])
                eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1])

                next_observation, reward, done = self.interpolation_step(env, state0, [throttle_action, eng_ctrl_action])
                state0 = next_observation[-1:][0]

                exp_buffer.store(actor_hx, observation, [throttle_action, eng_ctrl_action], reward, float(done))

                if (epoch_steps % (self.data_send_steps + self.N) == 0 and epoch_steps > 0) or done:
                    self.prepare_and_send_replay_data(exp_buffer, self.data_send_steps)

                if global_step % self.exchange_steps == 0 and self.training_active.value > 0: # update target networks every 'exchange_steps'
                    self.get_target_weights()
                    
                observation = next_observation
                global_step+=1
                epoch_steps+=1
                episodic_reward += reward

            rewards_history.append(episodic_reward)
            last_mean = np.mean(rewards_history[-100:])
            print(f'[{self.id} {i} ({epoch_steps})] Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
            if last_mean > 200:
                self.actor.save(f'lunar_lander_R2D2_{self.id}.h5')
                break
        env.close()
        print(f'Agent [{self.id}] done training.')

def RunActor(id:int, gamma:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = Actor(id, gamma, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()
