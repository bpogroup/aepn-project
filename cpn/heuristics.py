from random import random


class SPT:
    """
    This class acts as a shortest processing time heuristic for the AEPN_Env environment with graph observations
    """
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        """
        Returns the action with the shortest processing time
        """
        #get all neighbors of 'a' transitions nodes in observation
        #import pdb; pdb.set_trace()
        bindings = observation['actions_dict']

        min_time = float('inf')
        max_budget = float('-inf')
        selected_key = None

        for key, value in bindings.items():
            for action in value[0]:
                #import pdb; pdb.set_trace()
                if action[0].src._id == 'resources':
                    avg_completion_time = action[1].color.get('average_completion_time', 0)
                elif action[0].src._id == 'waiting':
                    budget = action[1].color['budget']

            if avg_completion_time < min_time or (avg_completion_time == min_time and budget > max_budget):
                min_time = avg_completion_time
                max_budget = budget
                selected_key = key

        return selected_key