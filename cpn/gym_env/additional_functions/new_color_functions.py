import json

import numpy
import numpy as np
import math
import random

def check_compatibility(c1, c2):
    c1 = c1.split(';')
    c2 = c2.split(';')
    if set(c1) & set(c2):
        return True
    return False

def combine(c1, c2):
    """
    Compose the colors lists of colors (which is a list of colors) and sets the result as the current color_list
    """
    return c1 + '|' + c2

def split(color, index):
    """
    Splits a combined color in its original components and returns the component at the given index
    """
    return color.split('|')[index]

def f(x):
    return x

def check_remaining_space(object, bin):
    """
    Checks if the remaining space in bin is enough to contain the object
    """
    obj_dict = json.loads(object)
    bin_dict = json.loads(bin)

    if int(bin_dict["curr_level"]) + int(obj_dict["weight"]) <= int(bin_dict["capacity"]):
        return True
    return False

def assign_obj_to_bin(object, bin):
    """
    Assigns object to bin by decreasing the space left in the bin by the space of the object
    """
    obj_dict = json.loads(object)
    bin_dict = json.loads(bin)

    bin_dict['curr_level'] += obj_dict['weight'] #or curr_level, indifferently
    
    return json.dumps(bin_dict).replace(" ", "")

def empty(bin):
    bin_dict = json.loads(bin)

    bin_dict['curr_level'] = 0 #or curr_level, indifferently
    
    return json.dumps(bin_dict).replace(" ", "")

def percent_full(bin):
    """
    Reward function based on the percentage of fullness of every bin
    """
    bin_dict = json.loads(bin)

    return float(bin_dict["curr_level"])/float(bin_dict["capacity"])


def check_adjacency(agent, position):
    """
    Check if the agent is adjacent to the given position
    """
    agent_dict = json.loads(agent)
    position_dict = json.loads(position)
    return (abs(agent_dict['x'] - position_dict['x']) <= 1 and agent_dict['y'] == position_dict['y']) or (abs(agent_dict['y'] - position_dict['y']) <= 1 and agent_dict['x'] == position_dict['x'])

def check_zero_ttl(object, true_if_zero):
    """
    Check if the object has reached a ttl of zero
    """
    object_dict = json.loads(object)
    if object_dict['ttl'] == 0:
        return true_if_zero
    else:
        return not true_if_zero

def check_coordinates(agent, object):
    """
    Check if the coordinated of agent are the same as those of object
    """
    agent_dict = json.loads(agent)
    object_dict = json.loads(object) 
    return agent_dict['x'] == object_dict['x'] and agent_dict['y'] == object_dict['y'] 

def decrement_ttl(object):
    """
    Decrements object's ttl by 1
    """
    object_dict = json.loads(object)

    object_dict['ttl'] -= 1

    return json.dumps(object_dict).replace(" ", "")

def soft_compatibility(task, res):
    """
    Defines token's delay according to a soft compatibility rule (see colors in task_assignment_soft_comp.txt)
    """
    res_dict = json.loads(res) 
    return res_dict[task]

def check_leaving(dummy_res):
    """
    Checks if the dummy resource is leaving (or returning)
    """
    dummy_res_dict = json.loads(dummy_res)
    return dummy_res_dict['leaving']

def set_random_leaving(dummy_res):
    """
    At each step, there is a 50% pobability that the resource will leave (or return)
    """
    is_leaving = np.random.randint(2)
    if bool(is_leaving):
        dummy_res = switch_boolean(dummy_res)
    return dummy_res

def switch_boolean(dummy_res):
    """
    Switches the dummy resource's boolean value
    """
    dummy_res_dict = json.loads(dummy_res)
    dummy_res_dict['leaving'] = int(not bool(dummy_res_dict['leaving']))
    return json.dumps(dummy_res_dict).replace(" ", "")

def get_res(dummy_res):
    """
    Returns the dummy resource
    """
    dummy_res_dict = json.loads(dummy_res)
    return dummy_res_dict['res']

def check_rew(task):
    """
    Returns the reward of the task
    """
    task_dict = json.loads(task)
    return task_dict['rew']

def decrement_n_res(task):
    """
    Decrements the number of resources required for the task
    """
    task_dict = json.loads(task)
    task_dict['n_res'] -= 1
    return json.dumps(task_dict).replace(" ", "")

def check_n_res(task, true_if_zero=True):
    """
    Checks if the task is completed
    """
    object_dict = json.loads(task)
    if object_dict['n_res'] == 0:
        return true_if_zero
    else:
        return not true_if_zero

def is_not_set(resource):
    """
    Checks if the dummy resource is not set
    """
    #import pdb; pdb.set_trace()
    return not bool(resource['is_set'])

def set_compatibility(resource):
    """
    Sets the resource's compatibilities
    """
    if resource['type']==0:
        #generate a random compatibility
        resource['compatibility_0'] = 1 + np.random.randint(2)
        resource['compatibility_1'] = 3 + np.random.randint(2)
    elif resource['type']==1:
        resource['compatibility_0'] = 3 + np.random.randint(2)
        resource['compatibility_1'] = 1 + np.random.randint(2)

    resource['is_set'] = True
    return {'type': resource['type'], 'compatibility_0': resource['compatibility_0'], 'compatibility_1': resource['compatibility_1'], 'is_set': resource['is_set']}

def new_combine(case, resource):
    return {'resource_type': resource['type'], 'task_type': case['type'], 'compatibility_0': resource['compatibility_0'], 'compatibility_1' : resource['compatibility_1']}

def new_split(case_resource):
    return {'type': case_resource['resource_type'], 'compatibility_0': case_resource['compatibility_0'], 'compatibility_1': case_resource['compatibility_1'], 'is_set': False}

def get_reward(case, resource):
    if case['type'] == resource['type']:
        return 10
    else:
        return 0

def get_budget_str(case, resource):
    case = json.loads(case)
    return case['budget']

def get_budget(case, resource):
    return case['budget']

#NOTE: CLOCK is a reserved name for the clock variable, and it must be passed as the first argument to the function
def is_late(CLOCK, case):
    if case['patience'] <= CLOCK:
       print('Late case')
    return case['patience'] <= CLOCK


def combine_budget(case, resource):
    return {'task_type':case['type'], 'resource_average_completion_time': resource['average_completion_time'], 'budget': case['budget']}

def get_resource(case_resource):
    return {'average_completion_time': case_resource['resource_average_completion_time']}

def randomize_budget_and_patience(CLOCK, x):
    x = x.copy()
    del x['average_interarrival_time']
    if not 'type' in x.keys():
        mu = 100  # Mean budget
        sigma = 10  # Standard deviation budget


    else:
        if x['type'] == 0:
            mu = 100
        else:
            mu = 200
        sigma = 10

    x['budget'] = round(random.gauss(mu, sigma), 2)  # budget is rounded to 2 decimal places

    #import pdb; pdb.set_trace()
    x['patience'] = CLOCK + random.expovariate(1/2) #the interarrival time is 1, so the patience should be on average higher than the arrivals
    x['arrival_time'] = CLOCK
    return x


def randomize_budget(x):
    x = x.copy() #CAREFUL: if x is a dictionary, it is passed by reference, so it is modified in place. In general, we want to make a copy at the beginning of the function
    if x['type'] == 0:
        mu = 100
    else:
        mu = 200
    sigma = 25  # Standard deviation budget
    x['budget'] = round(random.gauss(mu, sigma), 2) #budget is an integer

    return x

def randomize_budget_uniform(x):
    x = x.copy()
    if x['type'] == 0:
        x['budget'] = round(numpy.random.uniform(70, 130), 2)
    else:
        x['budget'] = round(numpy.random.uniform(170, 230), 2)
    return x

def randomize_budget_uniform_str(x):
    x = json.loads(x)
    if x['type'] == 0:
        x['budget'] = round(numpy.random.uniform(70, 130), 2)
    else:
        x['budget'] = round(numpy.random.uniform(170, 230), 2)
    return json.dumps(x).replace(" ", "")
def set_fixed_budget(x):
    x = x.copy()
    if x['type'] == 0:
        x['budget'] = 100
    else:
        x['budget'] = 200
    return x

def empty_token(case_resource):
    return {}
