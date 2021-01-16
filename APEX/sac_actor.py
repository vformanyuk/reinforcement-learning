import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from APEX.neural_networks import sac_policy_network, sac_critic_network
from APEX.APEX_Local_MemoryBuffer import APEX_NStepReturn_MemoryBuffer

class Actor(object):
    def __init__(self, id:int, gamma:float,
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value,
                 *args, **kwargs):
        # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
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
        self.tau = 0.01 #1 / self.exchange_steps #0.001
        self.gamma = gamma
        self.gaus_distr = tfp.distributions.Normal(0,1)
        self.N = 5
        self.action_bounds_epsilon = 1e-6
        self.log_std_min = -20
        self.log_std_max = 2
        self.alpha_log = 0.5

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.actor.set_weights(weights[0])
            self.critic1.set_weights(weights[1])
            self.critic2.set_weights(weights[2])
            self.alpha_log = float(weights[3])
            self.log(f'Target actor and target critic weights refreshed.')
        except EOFError:
            print("[get_target_weights] Connection closed.")
        except OSError:
            print("[get_target_weights] Connection closed.")

    def send_replay_data(self, states, actions, next_states, rewards, gamma_powers, dones, td_errors):
        buffer = []
        for i in range(len(states)):
            buffer.append([states[i], actions[i], next_states[i], rewards[i], gamma_powers[i], dones[i], td_errors[i]])
        try:
            self.cmd_pipe.send([1, self.id])
            self.log(f'Replay data command sent.')
            self.replay_pipe.send(buffer)
            self.log(f'Replay data sent.')
        except EOFError:
            print("[send_replay_data] Connection closed.")
        except OSError:
            print("[send_replay_data] Connection closed.")

    def __soft_update_models(self):
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

    def __prepare_and_send_replay_data(self, exp_buffer:APEX_NStepReturn_MemoryBuffer, batch_length:int):
        states, actions, next_states, rewards, gamma_powers, dones, _ = exp_buffer.get_tail_batch(batch_length)
        td_errors = self.get_td_errors(states, actions, next_states, rewards, gamma_powers, dones)
        self.send_replay_data(states, actions, next_states, rewards, gamma_powers, dones, td_errors)

    @tf.function(experimental_relax_shapes=True)
    def get_actions(self, mu, log_sigma, noise=None):
        if noise is None:
            noise = self.gaus_distr.sample()
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

    @tf.function(experimental_relax_shapes=True)
    def get_log_probs(self, mu, sigma, actions):
        action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        z = self.gaus_distr.sample()
        # appendix C of the SAC paper discribe applyed boundings which is log(1-tanh(u)^2)
        log_probs = action_distributions.log_prob(mu + sigma*z) - \
                    tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + self.action_bounds_epsilon), axis=1)
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def get_td_errors(self, states, actions, next_states, rewards, gamma_powers, dones):
        mu, log_sigma = self.actor(next_states, training=False)
        mu = tf.squeeze(mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), self.log_std_min, self.log_std_max)

        target_actions = self.get_actions(mu, log_sigma)

        min_target_q = tf.math.minimum(self.target_critic1([next_states, target_actions], training=False), \
                                       self.target_critic2([next_states, target_actions], training=False))

        sigma = tf.math.exp(log_sigma)
        log_probs = self.get_log_probs(mu, sigma, target_actions)
        next_values = tf.squeeze(min_target_q, axis=1) - tf.math.exp(self.alpha_log) * log_probs # min(Q1^,Q2^) - alpha * logPi

        target_q = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * next_values

        min_q = tf.math.minimum(self.critic1([states, actions], training=False), \
                                self.critic2([states, actions], training=False))

        return tf.math.abs(target_q - tf.squeeze(min_q, axis=1))

    def run(self):
        # this configuration must be done for every actor
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #try:
        #    tf.config.experimental.set_memory_growth(gpus[0], True)
        #    assert tf.config.experimental.get_memory_growth(gpus[0])
        #except:
        #    pass

        env = gym.make('LunarLanderContinuous-v2')
        self.actor = sac_policy_network((env.observation_space.shape[0]), env.action_space.shape[0])
        
        self.critic1 = sac_critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_critic1 = sac_critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.critic2 = sac_critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_critic2 = sac_critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.get_target_weights()

        exp_buffer = APEX_NStepReturn_MemoryBuffer(1001, self.N, self.gamma, env.observation_space.shape, env.action_space.shape)
        rewards_history = []
        
        global_step = 0
        for i in range(50000):
            if self.cancelation_token.value != 0:
                break
            
            done = False
            observation = env.reset()

            exp_buffer.reset()
            #data_send_step = 0

            episodic_reward = 0
            epoch_steps = 0
            critic_loss_history = []
            actor_loss_history = []

            while not done and self.cancelation_token.value == 0:
                mean, log_std_dev = self.actor(np.expand_dims(observation, axis = 0), training=False)
                throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0])
                eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1])

                next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])
                exp_buffer.store(observation, [throttle_action, eng_ctrl_action], next_observation, reward, float(done))

                if (epoch_steps % (self.data_send_steps + self.N) == 0 and epoch_steps > 0) or done:
                    self.__prepare_and_send_replay_data(exp_buffer, self.data_send_steps)
                    #data_send_step+=1

                if global_step % self.exchange_steps == 0 and self.training_active.value > 0: # update target networks every 'exchange_steps'
                    self.get_target_weights()

                if global_step % 10 == 0:
                    self.__soft_update_models()
                    
                observation = next_observation
                global_step+=1
                epoch_steps+=1
                episodic_reward += reward

            # don't forget to send terminal states
            #last_data_len = epoch_steps - data_send_step * (self.data_send_steps + self.N)
            #if last_data_len > 0 and self.cancelation_token.value == 0:
            #    self.__prepare_and_send_replay_data(exp_buffer, last_data_len)

            rewards_history.append(episodic_reward)
            last_mean = np.mean(rewards_history[-100:])
            print(f'[{self.id} {i} ({epoch_steps})] Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
            if last_mean > 200:
                self.actor.save(f'lunar_lander_apex_dpg_{self.id}.h5')
                break
        env.close()
        print(f'Agent [{self.id}] done training.')


def RunActor(id:int, gamma:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = Actor(id, gamma, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()
