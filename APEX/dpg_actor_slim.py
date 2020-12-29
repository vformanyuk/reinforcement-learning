import gym
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from APEX.neural_networks import policy_network, critic_network
from APEX.APEX_Local_MemoryBuffer import APEX_NStepReturn_RandomAccess_MemoryBuffer
from rl_utils.OUActionNoise import OUActionNoise

class ActorSlim(object):
    def __init__(self, id:int, batch_size:int, gamma:float, actor_leraning_rate:float, critic_learning_rate:float,
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
        self.tau = 0.1
        self.gamma = gamma
        self.N = 3

    def log(self, msg):
        if self.debug_mode:
            print(f'[Actor {self.id}] {msg}')

    def get_target_weights(self):
        try:
            self.cmd_pipe.send([0, self.id])
            weights = self.weights_pipe.recv()
            self.target_actor.set_weights(weights[0])
            self.target_critic.set_weights(weights[1])
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
        target_actor_weights = self.target_actor.get_weights()
        actor_weights = self.actor.get_weights()
        updated_actor_weights = []
        for aw,taw in zip(actor_weights,target_actor_weights):
            updated_actor_weights.append(self.tau * aw + (1.0 - self.tau) * taw)
            #updated_actor_weights.append(self.tau * taw + (1.0 - self.tau) * aw) #reversed
        self.actor.set_weights(updated_actor_weights) #target_actor

        target_critic_weights = self.target_critic.get_weights()
        critic_weights = self.critic.get_weights()
        updated_critic_weights = []
        for cw,tcw in zip(critic_weights,target_critic_weights):
            updated_critic_weights.append(self.tau * cw + (1.0 - self.tau) * tcw)
            #updated_critic_weights.append(self.tau * tcw + (1.0 - self.tau) * cw) #reversed
        self.critic.set_weights(updated_critic_weights) #target_critic

    @tf.function
    def get_td_errors(self, states, actions, next_states, rewards, gamma_powers, dones):
        target_mu = self.target_actor(next_states, training=False)
        target_q = rewards + tf.math.pow(self.gamma, gamma_powers + 1) * tf.reduce_sum((1 - dones) * self.target_critic([next_states, target_mu], training=False), axis = 1)
        current_q = tf.squeeze(self.critic([states, actions], training=False), axis=1)
        return tf.math.abs(target_q - current_q)

    def run(self):
        # this configuration must be done for every module
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        env = gym.make('LunarLanderContinuous-v2')
        self.actor = policy_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_actor = policy_network((env.observation_space.shape[0]), env.action_space.shape[0])
        
        self.critic = critic_network((env.observation_space.shape[0]), env.action_space.shape[0])
        self.target_critic = critic_network((env.observation_space.shape[0]), env.action_space.shape[0])

        self.get_target_weights()

        action_noise = OUActionNoise(mu=np.zeros(env.action_space.shape[0]))

        exp_buffer = APEX_NStepReturn_RandomAccess_MemoryBuffer(128, self.N, self.gamma, env.observation_space.shape, env.action_space.shape)
        rewards_history = []
        global_step = 0
        for i in range(50000):
            if self.cancelation_token.value != 0:
                break
            
            done = False
            observation = env.reset()

            episodic_reward = 0
            epoch_steps = 0
            critic_loss_history = []
            actor_loss_history = []

            while not done and self.cancelation_token.value == 0:
                chosen_action = self.actor(np.expand_dims(observation, axis = 0), training=False)[0].numpy() + action_noise()
                next_observation, reward, done, _ = env.step(chosen_action)
                exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

                if global_step % (self.data_send_steps + self.N) == 0 and global_step > 0:
                    states, actions, next_states, rewards, gamma_powers, dones, _ = exp_buffer.get_transfer_data(self.data_send_steps)
                    td_errors = self.get_td_errors(states, actions, next_states, rewards, gamma_powers, dones)
                    self.send_replay_data(states, actions, next_states, rewards, gamma_powers, dones, td_errors)

                if global_step % self.exchange_steps == 0 and self.training_active.value > 0: # update target networks every 'exchange_steps'
                    self.get_target_weights()
                    self.__reverse_soft_update_models()

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


def RunActor(id:int, batch_size:int, gamma:float, actor_leraning_rate:float, critic_learning_rate:float,
             cmd_pipe:mp.Pipe, weights_pipe:mp.Pipe, replay_data_pipe:mp.Pipe, cancelation_token:mp.Value, training_active_flag:mp.Value):
    actor = ActorSlim(id, batch_size, gamma, actor_leraning_rate, critic_learning_rate, cmd_pipe, weights_pipe, replay_data_pipe, cancelation_token, training_active_flag)
    actor.run()