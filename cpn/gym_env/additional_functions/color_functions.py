import json
import numpy as np

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
    if is_leaving:
        dummy_res = switch_boolean(dummy_res)
    return dummy_res

def switch_boolean(dummy_res):
    """
    Switches the dummy resource's boolean value
    """
    dummy_res_dict = json.loads(dummy_res)
    dummy_res_dict['leaving'] = int(not dummy_res_dict['leaving'])
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

