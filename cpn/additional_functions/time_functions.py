import random


def get_compatibility(case, resource):
    """
    Sets the resource's compatibilities
    """
    if case['type'] == 0:
        increment = resource['compatibility_0']
    elif case['type'] == 1:
        increment = resource['compatibility_1']
    else:
        raise Exception('Task type not recognized')

    return increment

def exponential_mean(x):
    mean = x['average_interarrival_time']
    return random.expovariate(1/mean)

def randomize_completion_time(case, resource):

    return resource['average_completion_time'] #not random for now
