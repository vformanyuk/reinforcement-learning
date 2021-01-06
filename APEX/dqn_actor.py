import gym
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from APEX.neural_networks import q_network
from APEX.APEX_Local_MemoryBuffer import APEX_NStepReturn_MemoryBuffer

class DQNActor(object):
    def __init__(self, id:int, gamma:float, epsilon:float,
                 cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value,
                 *args, **kwargs):
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
        self.epsilon = epsilon
        self.N = 5

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.target_Q.set_weights(weights[0])
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

    def __reverse_soft_update_models(self):
        target_weights = self.target_Q.get_weights()
        q_weights = self.main_Q.get_weights()
        updated_weights = []
        for aw,taw in zip(q_weights,target_weights):
            #updated_actor_weights.append(self.tau * aw + (1.0 - self.tau) * taw)
            updated_weights.append(self.tau * taw + (1.0 - self.tau) * aw) #reversed
        self.main_Q.set_weights(updated_weights) #target_actor

    def __prepare_and_send_replay_data(self, exp_buffer:APEX_NStepReturn_MemoryBuffer, batch_length:int):
        states, actions, next_states, rewards, gamma_powers, dones, _ = exp_buffer.get_tail_batch(batch_length)
        td_errors = self.get_td_errors(states, actions, next_states, rewards, gamma_powers, dones)
        self.send_replay_data(states, actions, next_states, rewards, gamma_powers, dones, td_errors)

    @tf.function(experimental_relax_shapes=True)
    def get_td_errors(self, states, actions, next_states, rewards, gamma_powers, dones):
        one_hot_actions_mask = tf.one_hot(actions, depth=self.action_space_dims, on_value = 1.0, off_value = 0.0, dtype=tf.float32) #shape batch_size,4

        target_Q_values = self.target_Q(next_states, training=False) # shape = (batch_size,4)
        target_y = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * tf.reduce_max(target_Q_values, axis=1) * (1 - dones) # shape = (batch_size,)
    
        pred_Q_values = self.main_Q(states, training=False) # shape = (batch_size,4)
        pred_y = tf.reduce_sum(tf.math.multiply(pred_Q_values, one_hot_actions_mask), axis=1) # Q values for non-chosen action do not impact loss. shape = (batch_size,)

        return tf.math.abs(target_y - pred_y)

    def epsilon_greedy(self, env, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(env.action_space.n)
        else:
            #q_actions =  self.main_Q.predict(np.expand_dims(observation, axis = 0))
            q_actions = self.main_Q(np.expand_dims(observation, axis = 0), training=False)[0].numpy()
            return np.argmax(q_actions)

    def run(self):
        # this configuration must be done for every module
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        env = gym.make('LunarLander-v2')
        self.main_Q = q_network((env.observation_space.shape[0]), env.action_space.n)
        self.target_Q = q_network((env.observation_space.shape[0]), env.action_space.n)
        self.action_space_dims = env.action_space.n

        self.get_target_weights()

        exp_buffer = APEX_NStepReturn_MemoryBuffer(1001, self.N, self.gamma, env.observation_space.shape, env.action_space.shape, action_type = np.int32)
        rewards_history = []
        
        global_step = 0
        for i in range(50000):
            if self.cancelation_token.value != 0:
                break
            
            done = False
            observation = env.reset()

            exp_buffer.reset()
            data_send_step = 0

            episodic_reward = 0
            epoch_steps = 0
            critic_loss_history = []
            actor_loss_history = []

            while not done and self.cancelation_token.value == 0:
                chosen_action = self.epsilon_greedy(env, observation)
                next_observation, reward, done, _ = env.step(chosen_action)
                exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

                if epoch_steps % (self.data_send_steps + self.N) == 0 and epoch_steps > 0:
                    self.__prepare_and_send_replay_data(exp_buffer, self.data_send_steps)
                    data_send_step+=1

                if global_step % self.exchange_steps == 0 and self.training_active.value > 0: # update target networks every 'exchange_steps'
                    self.get_target_weights()

                if global_step % 10 == 0:
                    self.__reverse_soft_update_models()
                    
                observation = next_observation
                global_step+=1
                epoch_steps+=1
                episodic_reward += reward

            # don't forget to send terminal states
            last_data_len = epoch_steps - data_send_step * (self.data_send_steps + self.N)
            if last_data_len > 0:
                self.__prepare_and_send_replay_data(exp_buffer, last_data_len)

            rewards_history.append(episodic_reward)
            last_mean = np.mean(rewards_history[-100:])
            print(f'[{self.id} {i} ({epoch_steps})] Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
            if last_mean > 200:
                self.actor.save(f'lunar_lander_apex_dpg_{self.id}.h5')
                break
        env.close()


def RunActor(id:int, gamma:float, epsilon:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = DQNActor(id, gamma, epsilon, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()
