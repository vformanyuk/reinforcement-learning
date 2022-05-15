import numpy as np
import tensorflow as tf

class Trajectory:
    def __init__(self, N:int, trajectory_length:int, burn_in_length:int):
        self.burn_in = np.zeros(burn_in_length, dtype=np.int32) if burn_in_length > 0 else []
        self.data = np.zeros(trajectory_length, dtype=np.int32)
        self.write_idx = 0
        self.writing_burn_in = burn_in_length > 0
        self.n_backup = N

    def add(self, idx):
        if self.is_complete():
            return

        if self.writing_burn_in:
            self.burn_in[self.write_idx] = idx
        else:
            self.data[self.write_idx] = idx

        self.write_idx += 1
        if self.writing_burn_in and self.write_idx == len(self.burn_in):
            self.writing_burn_in = False
            self.data[0] = self.burn_in[-1]
            self.write_idx = 1

    def is_complete(self) -> bool:
        return self.write_idx >= len(self.data)

    def is_n_backup_complete(self, buffer_idx) -> bool:
        return self.write_idx >= len(self.data) and self.data[-1] + self.n_backup >= buffer_idx

    def fix_trajectory(self):
        if self.writing_burn_in:
            return
        first_zero = -1
        for i,d in enumerate(self.data):
            if d == 0:
                first_zero = i
                break
        if first_zero == 1: # trajectory lenght must be at least 2. If it's less then take one more record from burn in
            self.data[1] = self.burn_in[-1]
            self.data[0] = self.burn_in[-2]
            first_zero += 1
        self.data = self.data[:first_zero] # truncate trailing zeros

class R2D2_AgentBuffer(object):
    def __init__(self, distributed_mode:bool, buffer_size:int, N:int, gamma:float, 
                state_shape, action_shape, trajectory_ready_callback = None, reward_shape=None, action_type = np.float32,
                trajectory_size=80, burn_in_length=10, trajectory_overlap = 40):
        self._distributed_mode = distributed_mode # when NOT in distributed mode trajectory cache and burn-in memory not cleared after episode end
        self._trajectory_size = trajectory_size
        self._trajectory_overlap = trajectory_overlap
        self._burn_in_len = burn_in_length
        self.buffer_size = buffer_size
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        real_reward_shape = (buffer_size,) if reward_shape == None else (buffer_size, *reward_shape)
        self.rewards_memory = np.empty(shape=real_reward_shape, dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.actor_hidden_states_memory = []
        self.trajectories = []
        self.trajectories.append(Trajectory(N, self._trajectory_size,0))
        self.sent_trajectories = []
        self.trajectory_ready = trajectory_ready_callback
        self.memory_idx = 0
        self.N = N
        self.gammas=[]
        self.gammas.append(1) # gamma**0
        self.gammas.append(gamma) #gamma**1
        for i in range(2, N + 1):
            self.gammas.append(np.power(gamma,i))

    def __len__(self):
        return len(self.trajectories)

    '''
    Pack (state, action, reward) tuples into solid trajectory. 
    Also produces burn-in trajectory that preceeds maion trajectory
    Note: Shouldn't create burn-in for first trajectory. Also, hidden state is 0 for first trajectory
    '''
    def store(self, actor_hidden_state:tf.Tensor, state:tf.Tensor, action:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        self.states_memory[self.memory_idx] = state
        self.actions_memory[self.memory_idx] = action
        self.rewards_memory[self.memory_idx] = 0
        self.gamma_power_memory[self.memory_idx] = 0
        self.dones_memory[self.memory_idx] = is_terminal
        self.actor_hidden_states_memory.append(actor_hidden_state)
        #propogate back current reward
        n_return_idx = 0
        while self.memory_idx - n_return_idx >= 0 and n_return_idx < self.N: # [0 .. N-1]
            self.rewards_memory[self.memory_idx - n_return_idx] += reward * self.gammas[n_return_idx]
            self.gamma_power_memory[self.memory_idx - n_return_idx] = n_return_idx
            n_return_idx += 1

        for idx, trajectory in enumerate(self.trajectories):
            trajectory.add(self.memory_idx)
            if self.trajectory_ready != None and idx not in self.sent_trajectories and trajectory.is_n_backup_complete(self.memory_idx):
                self.trajectory_ready(self, idx)
                self.sent_trajectories.append(idx)

        if self.memory_idx > 0 and \
            self.memory_idx % (self._trajectory_size - self._burn_in_len - self._trajectory_overlap) == 0 and \
            not is_terminal > 0:
            self.trajectories.append(Trajectory(self.N, self._trajectory_size, self._burn_in_len))

        self.memory_idx += 1

    def reset(self):
        if self._distributed_mode: # in distribured mode (for APE-X or R2D2) memory completly cleared after every episode
            self.actor_hidden_states_memory.clear()
            self.sent_trajectories.clear()
            self.trajectories.clear()
            self.trajectories.append(Trajectory(self.N, self._trajectory_size, 0))
        self.memory_idx = 0

    def get_remaining_trajectories(self):
        for idx, trajectory in enumerate(self.trajectories):
            if idx not in self.sent_trajectories and not trajectory.writing_burn_in:
                trajectory.fix_trajectory()
                yield idx
                if not trajectory.is_complete(): # stop producing trajectories after first incomplete met
                    break

    def get_data(self, trajectory_idxs):
        for idx in trajectory_idxs:
            hidden_state_idx = self.trajectories[idx].burn_in[0] if len(self.trajectories[idx].burn_in) > 0 else self.trajectories[idx].data[0]
            burn_in_idxs = self.trajectories[idx].burn_in
            states_idxs = self.trajectories[idx].data[:-1]
            trajectory_idxs = self.trajectories[idx].data[1:]
            burn_in_states = tf.stack(self.states_memory[burn_in_idxs]) if len(burn_in_idxs) > 0 else []
            burn_in_actions = tf.stack(self.actions_memory[burn_in_idxs]) if len(burn_in_idxs) > 0 else []
            yield self.actor_hidden_states_memory[hidden_state_idx], \
                    burn_in_states, \
                    burn_in_actions, \
                    tf.stack(self.states_memory[states_idxs]), \
                    tf.stack(self.actions_memory[trajectory_idxs]), \
                    tf.stack(self.states_memory[trajectory_idxs]), \
                    tf.stack(self.rewards_memory[trajectory_idxs]), \
                    tf.stack(self.gamma_power_memory[trajectory_idxs]), \
                    tf.stack(self.dones_memory[trajectory_idxs]), \
                    tf.stack([tf.squeeze(self.actor_hidden_states_memory[i], axis=0) for i in states_idxs])