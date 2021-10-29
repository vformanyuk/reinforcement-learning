class TransmitionBuffer:
    def __init__(self) -> None:
        self.actor_hx_mem=[]
        self.burn_in_mem=[]
        self.states_mem=[]
        self.actions_mem=[]
        self.next_states_mem=[]
        self.rewards_mem=[]
        self.gamma_powers_mem=[]
        self.dones_mem=[]
        self.iteration_idx = 0
    def append(self, actor_hx, burn_in, sates, actions, next_states, rewards, gps, dones):
        self.actor_hx_mem.append(actor_hx)
        self.burn_in_mem.append(burn_in)
        self.states_mem.append(sates)
        self.actions_mem.append(actions)
        self.next_states_mem.append(next_states)
        self.rewards_mem.append(rewards)
        self.gamma_powers_mem.append(gps)
        self.dones_mem.append(dones)
    def __len__(self):
        return len(self.rewards_mem)
    def __iter__(self):
        self.iteration_idx = -1
        return self
    def __next__(self):
        raise StopIteration

class AgentTransmitionBuffer(TransmitionBuffer):
    def __init__(self) -> None:
        super(AgentTransmitionBuffer, self).__init__()
        self.td_errors_mem=[]
    def append(self, actor_hx, burn_in, sates, actions, next_states, rewards, gps, dones, td_error):
        super(AgentTransmitionBuffer, self).append(actor_hx, burn_in, sates, actions, next_states, rewards, gps, dones)
        self.td_errors_mem.append(td_error)
    def __len__(self):
        return len(self.rewards_mem)
    def __next__(self):
        self.iteration_idx += 1
        if self.iteration_idx >= self.__len__():
            raise StopIteration
        return self.actor_hx_mem[self.iteration_idx], \
                self.burn_in_mem[self.iteration_idx], \
                self.states_mem[self.iteration_idx], \
                self.actions_mem[self.iteration_idx], \
                self.next_states_mem[self.iteration_idx], \
                self.rewards_mem[self.iteration_idx], \
                self.gamma_powers_mem[self.iteration_idx], \
                self.dones_mem[self.iteration_idx], \
                self.td_errors_mem[self.iteration_idx]

class LearnerTransmitionBuffer(TransmitionBuffer):
    def __init__(self) -> None:
        super(LearnerTransmitionBuffer, self).__init__()
        self.is_weights_mem = []
        self.meta_idxs_mem = []
    def append(self, actor_hx, burn_in, sates, actions, next_states, rewards, gps, dones, is_weights, meta_idxs):
        super(LearnerTransmitionBuffer, self).append(actor_hx, burn_in, sates, actions, next_states, rewards, gps, dones)
        self.is_weights_mem.append(is_weights)
        self.meta_idxs_mem.append(meta_idxs)
    def __len__(self):
        return len(self.rewards_mem)
    def __next__(self):
        self.iteration_idx += 1
        if self.iteration_idx >= self.__len__():
            raise StopIteration
        return self.actor_hx_mem[self.iteration_idx], \
                self.burn_in_mem[self.iteration_idx], \
                self.states_mem[self.iteration_idx], \
                self.actions_mem[self.iteration_idx], \
                self.next_states_mem[self.iteration_idx], \
                self.rewards_mem[self.iteration_idx], \
                self.gamma_powers_mem[self.iteration_idx], \
                self.dones_mem[self.iteration_idx], \
                self.is_weights_mem[self.iteration_idx], \
                self.meta_idxs_mem[self.iteration_idx]
