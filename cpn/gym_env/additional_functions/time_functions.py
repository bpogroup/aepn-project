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
