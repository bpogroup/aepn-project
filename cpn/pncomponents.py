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
    def __init__(self, _id, marking = dict(), colors_set = None):
        """
        A place has an identifier _id and a marking.
        The marking is represented as a dictionary token -> number of tokens (with that color and time).
        :param _id: the identifier of the place.
        """
        super().__init__(_id)
        self.marking = copy.deepcopy(marking)
        self.colors_set = copy.deepcopy(colors_set)

    def add_token(self, token, count=1):
        if token not in self.marking:
            self.marking[token] = count
        else:
            self.marking[token] += count

    def remove_token(self, token):
        if token in self.marking:
            self.marking[token] -= 1
        else:
            raise("No token '" + token + "' at place '" + str(self) + "'.")
        if self.marking[token] == 0:
            del self.marking[token]

    def marking_str(self):
        result = str(self) + ":"
        i = 0
        for (token, count) in self.marking.items():
            result += str(count) + "`" + str(token)
            if i < len(self.marking) - 1:
                result += "++"
            i += 1
        return result

class Transition(Node):

    def __init__(self, _id, guard=None):
        super().__init__(_id)
        self.guard = guard

    def set_guard(self, func):
        self.guard = func

class TaggedTransition(Transition):

    def __init__(self, _id, tag, reward, guard=None):
        super().__init__(_id, guard)
        self.tag = tag
        self.reward = reward

class Arc(AbstractArc):
    def __init__(self, src, dst, inscription, time_expression=None):
        self.src = src
        self.dst = dst
        self.inscription = inscription
        self.time_expression = time_expression

    def __str__(self):
        return f"({self.src}, {self.dst})"

    def __repr__(self):
        return f"({self.src}, {self.dst})"

class TimeIncreasingArc(Arc):
    #inf increment!=None, determistic increment
    def __init__(self, src, dst, inscription, increment = None, delay_type = None, params = None):
        super().__init__(src, dst, inscription)
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
    def __init__(self, color, time=None):
        self.color = color
        self.time = time

    def __eq__(self, token):
        return self.color == token.color and self.time == token.time

    def __lt__(self, token):
        return (self.color, self.time) < (token.color, token.time)

    def __hash__(self):
        return (self.color, self.time).__hash__()

    def __str__(self):
        result = str(self.color)
        if self.time is not None:
            result += "@" + str(self.time)
        return result

    def __repr__(self):
        return self.__str__()

class TimedToken(Token):
    def __init__(self, _id, color, time):
        super().__init__(_id, color)
        self.time = time

    def set_time(self, t):
        self.time = t

#TODO: make it a general class
class Color(AbstractColor):
    """
    Color is a dict cause it holds the arc that generated the token as key and the token's color as value
    """
    def __init__(self):
        self._id=uuid.uuid1()

    def __init__(self, color_dict={}):
        self._id=uuid.uuid1()
        self.color_dict = color_dict

    def check_compatibility(self, color):
        for l1 in list(self.color_dict.values()):
            for l2 in list(color.color_dict.values()):
                if set(l1) & set(l2):
                    return True
        return False

    def color_list(self):
        return list(self.color_dict.values())

    def __eq__(self, color):
        return self.color_list() == color.color_list()

    def __hash__(self):
        return hash(self._id)

    def __str__(self):
        return str(self.color_dict)

    def __repr__(self):
        return str(self.color_dict)

class AssociativeColor(Color):
    """
    Special color class to express associations between objects, such as tasks assigned to resources.
    Compatibility colors can be combined into a new color that holds the list of basic colors of each original color, and split back to the original colors.
    To combine colors a and b, the color_list of b is appended to the color list of a (thus obtaining a list of lists)
    """
    def __init__(self):
        super().__init__()
        self.color_list = []

    def __init__(self, color_list=[]):
        super().__init__()
        self.color_list.append(color_list)
        
    #TODO: make it work with an unbounded number of input colors
    @staticmethod
    def check_compatibility(colors_list):
        for l1 in colors_list[0].color_list:
            for l2 in colors_list[1].color_list:
                if set(l1) & set(l2):
                    return True
        return False

    @staticmethod
    def combine(colors):
        """
        Compose the colors lists of colors (which is a list of colors) and sets the result as the current color_list
        """
        color_list = []
        for c in colors:
            color_list.append(c.color_list)

        return color_list

    @staticmethod
    def split(color, index):
        """
        Splits a combined color in its original components and returns the component at the given index
        """
        return color.color_list[index]            

class CapacityColor(Color):
    """
    Special color class to express items' capacity (or weight)
    """
    def __init__(self):
        super().__init__()
        self.tot = None #total capacity
        self.left = None #available capacity

    def __init__(self, tot, left):
        super().__init__()
        self.tot = tot
        self.left = left

    #items that do not need to be filled with others do not hold a total capacity, but just a left capacity (that will be subtracted from the container's)
    def __init__(self, left):
        super().__init__()
        self.tot = None
        self.left = left
            
    def __eq__(self, color):
        return self.tot == color.tot and self.left == color.left

    def restore(self):
        self.left = self.tot
        
    def minus(self, color):
        self.left -= color.left
