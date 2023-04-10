from abc import ABC, abstractclassmethod
from gc import DEBUG_STATS
import copy
from numpy.random import default_rng
import json
import uuid
from .gym_env.additional_functions.time_functions import *


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
            result += str(count) + "`" + str(token) + "`"
            if i < len(self.marking) - 1:
                result += "++"
            i += 1
        return result
    
    def __str__(self):
        return self._id + f"({self.colors_set})"

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
        self._id = str(src) + '-' + str(dst)

    def __str__(self):
        return f"({self.src}, {self.dst})"

    def __repr__(self):
        return f"({self.src}, {self.dst})"

class TimeIncreasingArc(Arc):
    #if increment!=None, determistic increment
    def __init__(self, src, dst, inscription, increment = None, delay_type = None, params = None):
        super().__init__(src, dst, inscription)
        self.increment=increment
        self.delay_type = delay_type
        self.params = params
        self.rng = default_rng()

    def increment_time(self, token, clock):
        if self.increment == None:
            raise Exception(f"No increment was provided on arc {self}")
        elif type(self.increment) == int: #integer increment
            token.time = clock + self.increment
        elif type(self.increment) == str: #if a string is provided, it is assumed that it corresponds to an extra function defined in my_functions.py
            increment = 0
            f = open('./cpn/color_functions.py', 'r')
            additional_functions = f.read()
            f.close()
            exec(compile(additional_functions + "result = " + self.increment, "<string>", "exec"), increment)
            token.time = clock + increment
        #elif type(self.increment) == int: #random variable used as increment (currently, only geometric)
        #    if self.delay_type == 'geometric' and isinstance(self.params, int) and 0 <= self.params <= 100:
        #        token.time = clock + self.rng.geometric(float(self.params)/100)
        #        print(f"Updated token clock to value {token.time}")


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