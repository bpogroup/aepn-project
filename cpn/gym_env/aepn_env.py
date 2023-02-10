import gymnasium as gym
from gym import spaces, Env
from typing import List
import numpy as np
import copy

class AEPN_Env(Env):

	def __init__(self, aepn):
		"""
		The environment requires an A-E PN that serves as guide for the simulation
		"""
		super().__init__()
		self.pn = aepn #the petri net is used for taking actions only when tagged as 'a'
		self.frozen_pn = copy.deepcopy(self.pn)

		#a-e pn elements that are needed for defining actions and observations
		self.pn_actions = aepn.get_actions() #retrieve the color sets of places incoming to actions transitions
		pn_observations = aepn.get_observation() #retrieve, for each place, the amount of tokens of each color (if in vector mode)
		
		#gym specific
		self.observation_space = spaces.Box(shape=(len(pn_observations),), low = 0, high = np.iinfo(np.uint8).max, dtype=np.uint8) #observations are the state representation that is shown to the solving algorithm
		self.action_space = spaces.Discrete(len(self.pn_actions)) #actions are all the available decisions (invalid actions are defined on a single observation through a masking mechanism)
		
		#mimic the network's organization
		self.run = []
		self.i = 0
		self.active_model = True
		self.prev_tag = self.pn.tag

	def step(self, action):
        
		old_rewards = self.pn.rewards
		old_clock = self.pn.clock
		observation, loc_i = self.pn.apply_action(self.pn_actions[action])
		self.i += loc_i
		terminated = False

		while self.pn.tag == "e" and not terminated:
			observation, terminated, self.i = self.pn.run_evolutions(self.run, self.i, self.active_model)
			if terminated: print('Terminated!')
		
		reward = (self.pn.rewards - old_rewards)/(1+(self.pn.clock - old_clock))
		
		info = {'pn_reward' : self.pn.rewards}
		return observation, reward, terminated, info

	def reset(self):
		print("Entered reset \n")
		#return observation = [], info = {}
		self.pn = copy.deepcopy(self.frozen_pn)
		self.pn.get_to_first_action()
		return self.pn.get_observation()

	# define mask based on current environment observation
	def action_masks(self) -> List[bool]:
		v_a = self.pn.get_valid_actions()
		valid_actions = [True if i in v_a else False for i in self.pn_actions]
		self.valid_actions = valid_actions
		return valid_actions


    
        