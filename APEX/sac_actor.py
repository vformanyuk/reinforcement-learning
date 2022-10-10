import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing as mp

from APEX.neural_networks import sac_policy_network, sac_value_network
from APEX.APEX_Local_MemoryBuffer import APEX_NStepReturn_MemoryBuffer

class Actor(object):
    def __init__(self, id:int, gamma:float,
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value,
                 *args, **kwargs):
        # prevent TensorFlow of allocating whole GPU memory. Must be called in every module
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([], 'GPU')

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

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.actor.set_weights(weights[0])
            self.value_net.set_weights(weights[1])
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

    def __prepare_and_send_replay_data(self, exp_buffer:APEX_NStepReturn_MemoryBuffer, batch_length:int):
        states, actions, next_states, rewards, gamma_powers, dones, _ = exp_buffer.get_tail_batch(batch_length)
        td_errors = self.get_td_errors(states, next_states, rewards, gamma_powers, dones)
        self.send_replay_data(states, actions, next_states, rewards, gamma_powers, dones, td_errors)

    @tf.function
    def get_actions(self, mu, log_sigma, noise):
        return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

    @tf.function
    def get_td_errors(self, states, next_states, rewards, gamma_powers, dones):
        next_values = tf.squeeze(self.value_net(next_states, training=False), axis = 1)
        target_values = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * (1 - dones) * next_values
        current_values = tf.squeeze(self.value_net(states, training=False), axis = 1)
        return tf.math.abs(target_values - current_values)

    def run(self):
        env = gym.make('LunarLanderContinuous-v2')
        self.actor = sac_policy_network((env.observation_space.shape[0]), env.action_space.shape[0])        
        self.value_net = sac_value_network((env.observation_space.shape[0]))

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

            episodic_reward = 0
            epoch_steps = 0
            critic_loss_history = []
            actor_loss_history = []

            while not done and self.cancelation_token.value == 0:
                mean, log_std_dev = self.actor(np.expand_dims(observation, axis = 0), training=False)
                throttle_action = self.get_actions(mean[0][0], log_std_dev[0][0], self.gaus_distr.sample())
                eng_ctrl_action = self.get_actions(mean[0][1], log_std_dev[0][1], self.gaus_distr.sample())

                next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])
                exp_buffer.store(observation, [throttle_action, eng_ctrl_action], next_observation, reward, float(done))

                if (epoch_steps % (self.data_send_steps + self.N) == 0 and epoch_steps > 0) or done:
                    self.__prepare_and_send_replay_data(exp_buffer, self.data_send_steps)

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
                self.actor.save(f'lunar_lander_apex_dpg_{self.id}.h5')
                break
        env.close()
        print(f'Agent [{self.id}] done training.')


def RunActor(id:int, gamma:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = Actor(id, gamma, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()
