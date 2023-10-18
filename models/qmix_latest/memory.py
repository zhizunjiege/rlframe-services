from collections import namedtuple
import random
import numpy as np
Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','terminated'))

class ReplayMemory:
	def __init__(self,replay_size,agent_num,obs_dim,act_dim,dtype):
		self.capacity = replay_size
		self.memory = []
		self.position = 0
		self.size = 0
		self.agent_num = agent_num
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.dtype = dtype

	def store(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(Experience(*args))
		else :
			self.memory[self.position] = Experience(*args)
			self.position = int((self.position + 1)%self.capacity)
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)
	
	def get(self):
		state = {
			'agent_num': self.agent_num,
			'obs_dim': self.obs_dim,
			'act_dim': self.act_dim,
			'max_size': self.capacity,
			'dtype': self.dtype,
			'ptr': self.position,
			'size': self.size,
			'data': {
				'obs1_bufs': [self.memory[i].states for i in range(self.size)],
				'acts_bufs': [self.memory[i].actions for i in range(self.size)],
				'obs2_bufs': [self.memory[i].next_states for i in range(self.size)],
				'rews_buf': [self.memory[i].rewards for i in range(self.size)],
				'term_buf': [self.memory[i].terminated for i in range(self.size)],
			}
		}
		return state
	
	def set(self, buffer):
		s = buffer
		self.agent_num, self.max_size, self.dtype = s['agent_num'], s['max_size'], s['dtype']
		self.obs_dim, self.act_dim = s['obs_dim'], s['act_dim']
		self.position, self.size = s['ptr'], s['size']
		d = s['data']
		if len(self.memory)==0:
			for i in range(self.size):
				self.memory.append(Experience(d['obs1_bufs'][i], 
			       								d['acts_bufs'][i], 
												d['obs2_bufs'][i],
												d['rews_buf'][i],
												d['term_buf'][i]))
		else:
			for i in range(self.size):
				self.memory[i]._replace(states=d['obs1_bufs'][i],
			    						actions=d['acts_bufs'][i],
										next_states=d['obs2_bufs'][i],
										rewards=d['rews_buf'][i],
										terminated=d['term_buf'][i],)
