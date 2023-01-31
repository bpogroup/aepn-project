from abc import ABC, abstractclassmethod
from gc import DEBUG_STATS
import copy
from numpy.random import default_rng
import json
import uuid


class AbstractNode(ABC):
    @abstractclassmethod
    def __init__(self, id):
        pass

    @abstractclassmethod
    def __str__(self):
        pass

    @abstractclassmethod
    def __repr__(self):
        pass

    @abstractclassmethod
    def connect_arc(self,arc):
        pass

class AbstractColor(ABC):
    @abstractclassmethod
    def __init__(self, id):
        pass

    @abstractclassmethod
    def __str__(self):
        pass

class AbstractArc(ABC):
    @abstractclassmethod
    def __init__(self, id, ):
        pass

class AbstractToken(ABC):
    @abstractclassmethod
    def __init__(self):
        pass

class Node(AbstractNode):
    def __init__(self, _id):
        self._id = _id
        self.incoming = []
        self.outgoing = []

    def __str__(self):
        return self._id
    
    def __repr__(self):
        return self.__str__()

    def connect_arc(self, arc):
        if arc.src == self:
            self.outgoing.append(arc)
        elif arc.dst == self:
            self.incoming.append(arc)
        else:
            raise Exception(f"Trying to connect an arc that is not attached to node {self._id}")


class Place(Node):
    
    def __init__(self, _id, marking = [], colors_set = []):
        super().__init__(_id)
        self.colors_set = copy.deepcopy(colors_set) #TODO: implement in text-based representation
        self.marking = copy.deepcopy(marking)

    def __init__(self, _id):
        super().__init__(_id)
        self.colors_set = []
        self.marking = []

    def add_token(self, token):
        self.marking.append(token)

    def remove_token(self, token):
        self.marking.remove(token)

    def print_marking(self):
        ret_dict = {}
        for t in self.marking:
            if not str(t.color) in ret_dict.keys():
                ret_dict[str(t.color)] = 1
            else:
                ret_dict[str(t.color)] += 1
        return ret_dict


class Transition(Node):

    def __init__(self, _id):
        super().__init__(_id)
        self.guard_functions_list = []

    def set_guard(self, func):
        self.guard_functions_list.append(func)

class TaggedTransition(Transition):

    def __init__(self, _id, tag, reward):
        super().__init__(_id)
        self.tag = tag
        self.reward = reward

class Arc(AbstractArc):
    def __init__(self, src, dst, var):
        self.src = src
        self.dst = dst
        self.var = var

    def __str__(self):
        return f"({self.src}, {self.dst})"

    def __repr__(self):
        return f"({self.src}, {self.dst})"

class TimeWindowedArc(Arc):
    def __init__(self, src, dst, var, tw):
        super().__init__(src, dst, var)
        self.tw = tw

class TimeIncreasingArc(Arc):
    #inf increment!=None, determistic increment
    def __init__(self, src, dst, var, increment = None, delay_type = None, params = None):
        super().__init__(src, dst, var)
        self.increment=increment
        self.delay_type = delay_type
        self.params = params
        self.rng = default_rng()

    def increment_time(self, token, clock):
        if self.increment != None: #fixed increment
            token.time = clock + self.increment
        else: #random variable used as increment (currently, only geometric)
            if self.delay_type == 'geometric' and isinstance(self.params, int) and 0 <= self.params <= 100:
                token.time = clock + self.rng.geometric(float(self.params)/100)
                print(f"Updated token clock to value {token.time}")
            else:
                raise Exception("The specified delay distribution is not supported, or an error was made in the parameters' definition")

class Token:
    def __init__(self, _id, color):
        self._id=_id
        self.color = color

    def __eq__(self, token):
        return self.color == token.color

class TimedToken(Token):
    def __init__(self, _id, color, time):
        super().__init__(_id, color)
        self.time = time

    def set_time(self, t):
        self.time = t

#currently not used
class Color(AbstractColor):
    """
    Color is a dict cause it holds the arc that generated the token as key and the token's color as value
    """
    def __init__(self, color_dict={}):
        self._id=uuid.uuid1()
        self.color_dict = color_dict

    def __eq__(self, color):
        return self.color_list() == color.color_list() 

    def __hash__(self):
        return hash(self._id)

    def __str__(self):
        return str(self.color_dict)

    def __repr__(self):
        return str(self.color_dict)
            
    def check_compatibility(self, color):
        for l1 in list(self.color_dict.values()):
            for l2 in list(color.color_dict.values()):
                if set(l1) & set(l2):
                    return True
        return False

    def color_list(self):
        return list(self.color_dict.values())

    def compose_colors(self, colors):
        """
        Compose the colors lists of colors (which is a list of colors) and sets the result as the current color_list
        """
        pass

    def sum_colors(self, colors):
        """
        Sum the color values of colors (which is a list of colors) and sets the result as the current color_list
        """
        pass

