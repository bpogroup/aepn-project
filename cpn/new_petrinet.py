import random
import json
import itertools
import shutil
from abc import ABC, abstractclassmethod
import os
import logging

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.logger import configure
import copy
import time
from torch_geometric.data import Data, HeteroData

from graph.dynamic.train import make_parser, make_env, make_agent, make_logdir, HeteroActor

try:
    from .gym_env import new_aepn_env as aepn_env
    from .pncomponents import Transition, TaggedTransition, Place, Token, Arc
    from .gym_env.additional_functions.new_color_functions import *
except:
    from gym_env import new_aepn_env
    from pncomponents import Transition, TaggedTransition, Place, Token, Arc
    from gym_env.additional_functions.new_color_functions import *

class AbstractPetriNet(ABC):
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def __str__(self):
        pass

    @abstractclassmethod
    def add_place(self, place):
        pass

    @abstractclassmethod
    def add_transition(self, transition):
        pass

    @abstractclassmethod
    def add_arc(self, src, dst, var):
        pass

"""
Colored Petri Net implementation
"""
class PetriNet(AbstractPetriNet):
    def __init__(self, additional_functions):
        self.places = []
        self.transitions = []
        self.arcs = []
        self.clock = 0

        self.additional_functions = additional_functions
        if not additional_functions.endswith("\n"):
            self.additional_functions += "\n"
        compiled = compile(self.additional_functions, "<string>", 'exec')  # this is to test if the additional functions are well-formed
        if "result" in compiled.co_names:
            raise "'result' is a reserved name. It should not appear in the additional function definitions."

        self.by_id = dict()

    def __str__(self):
        result = ""
        result += "P={"
        for i in range(len(self.places)):
            result += str(self.places[i])
            if i < len(self.places) - 1:
                result += ","
        result += "}\n"
        result += "T={"
        for i in range(len(self.transitions)):
            result += str(self.transitions[i])
            if self.transitions[i].guard is not None:
                result += "[" + self.transitions[i].guard + "]"
            if i < len(self.transitions) - 1:
                result += ","
        result += "}\n"
        result += "A={"
        for i in range(len(self.arcs)):
            result += "(" + str(self.arcs[i].src) + "," + str(self.arcs[i].dst) + "," + str(self.arcs[i].inscription) + ")"
            if self.arcs[i].time_expression is not None:
                result += "@+" + str(self.arcs[i].time_expression) + ")"
            if i < len(self.arcs) - 1:
                result += ","
        result += "}\n"        
        result += "M={"
        for i in range(len(self.places)):
            result += self.places[i].marking_str()
            if i < len(self.places) - 1:
                result += ","
        result += "}"                
        return result

    def add_place(self, place):
        self.places.append(place)
        self.by_id[place._id] = place

    def add_transition(self, transition):
        self.transitions.append(transition)
        self.by_id[transition._id] = transition

    def add_arc(self, src, dst, inscription=None, time_expression=None):
        """
        Adds an arc to the Petri net from a source to a destination.
        An arc can have a variable by which a token value will be taken from the arcs incoming place (if any)
        or placed on the arcs outgoing place (if any).
        The variable can also be None (the default), in which case we assume the arc only processes 'black' tokens.
        :param src: a place or transition
        :param dst: a place or transition
        :param inscription: a string or None if there is no inscription.
        """
        if src not in self.places and src not in self.transitions:
            raise Exception("Attempting to create an arc from a non-existing source.")
        if dst not in self.places and dst not in self.transitions:
            raise Exception("Attempting to create an arc to a non-existing destination.")
        if src in self.places and dst in self.places:
            raise Exception("Attempting to create an arc from a place to a place.")
        if src in self.transitions and dst in self.transitions:
            raise Exception("Attempting to create an arc from a transition to a transition.")
        
        arc = Arc(src, dst, inscription, time_expression)
        self.arcs.append(arc)
        self.by_id[arc._id] = arc

        # TODO: there is probably a more elegant way to do this
        for n in self.places:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break
        for n in self.transitions:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break

    def add_arc_by_ids(self, src, dst, inscription, time_expression=0):
        if src not in self.by_id:
            raise Exception("Cannot find node with id '" + src + "' to construct arc to '" + dst + "'.")
        if dst not in self.by_id:
            raise Exception("Cannot find node with id '" + dst + "' to construct arc from '" + src + "'.")
        self.add_arc(self.by_id[src], self.by_id[dst], inscription=inscription, time_expression=time_expression)

    def add_mark(self, place, token):
        """
        Marks a place with a token.
        Each token has a value (can also be None to indicate a black token).
        :param place: a place
        :param tokens: a Token object
        """
        if not place in self.places:
            raise Exception("Attempting to mark a non-existing place.")
        place.add_token(token)

    def add_mark_by_id(self, place_id, color, time=0):
        """
        Marks a place with a token. The place is identified by its identifier.
        Each token has a value (can also be None to indicate a black token).
        :param place_id: a place identifier
        :param token: the token to be added
        :param count: the number of tokens with that color and time to be added (defaults to 1)
        """

        if place_id not in self.by_id:
            raise Exception("Cannot find place with id '" + place_id + "' to mark.")
        self.add_mark(self.by_id[place_id], Token(color, time))
        
    def tokens_combinations(self, transition):
        # create all possible combinations of incoming token values

        bindings = [[]]

        if self.observation_type == 'vector' or not self.observation_type:
            for arc in transition.incoming:
                new_bindings = []
                for token in arc.src.marking.keys(): #get set of colors in incoming place
                    for binding in bindings:
                        new_binding = binding.copy()
                        new_binding.append((arc, token))
                        new_bindings.append(new_binding)
                    bindings = new_bindings
        elif self.observation_type == 'graph':
            for arc in transition.incoming:
                new_bindings = []
                for token in arc.src.marking:
                    for binding in bindings:
                        new_binding = binding.copy()
                        new_binding.append((arc, token))
                        new_bindings.append(new_binding)
                bindings = new_bindings
        return bindings

    def transition_bindings(self, transition):
        """
        Calculates the set of bindings that enables the given transition.
        Each binding is a list of lists of (arc, token value) pairs, in which
        each list of pairs represents a single enabling binding.
        :param transition: the transition for which to calculate the enabling bindings.
        :param obs_type: used to define the type of binding (string-based for 'vector', object-based for 'graph')
        :return: a list of lists of (arc, token value) pairs
        """
        if len(transition.incoming) == 0:
            raise Exception("Though it is strictly speaking possible, we do not allow transitions like '" + str(self) + "' without incoming arcs.")

        bindings = self.tokens_combinations(transition)

        # a binding must have all incoming arcs
        nr_incoming_arcs = len(transition.incoming)
        new_bindings = []
        for binding in bindings:
            if len(binding) == nr_incoming_arcs:
                new_bindings.append(binding)
        bindings = new_bindings

        # if two arcs have matching arc variables they should also have matching values for a binding to be enabling
        result = []
        for binding in bindings:
            variable_values = dict()
            time = None
            enabled = True
            for (arc, token) in binding:
                if arc.inscription in variable_values:
                    if variable_values[arc.inscription] != token.color:
                        enabled = False
                        break
                variable_values[arc.inscription] = token.color
                if time is None or token.time > time:
                    time = token.time
            if enabled and transition.guard is not None:  # if the transition has a guard, that guard must evaluate to True with the given variable binding
                exec(compile(self.additional_functions + "result = " + transition.guard, "<string>", "exec"), variable_values)
                enabled = variable_values['result']
            if enabled:
                result.append((binding, time))
        return result

    def bindings(self):
        """
        Calculates the set of timed bindings that is enabled in the net.
        Each binding is a tuple ([(arc, token), (arc, token), ...], time) that represents a single enabling binding.
        If no timed binding is enabled at the current clock time, updates the current clock time to the earliest time at which there is.
        :return: list of tuples ([(arc, token), (arc, token), ...], time)
        """
        untimed_bindings = []
        timed_bindings = []
        for t in self.transitions:
            for (binding, time) in self.transition_bindings(t):
                if time is None:
                    untimed_bindings.append((binding, time))
                else:
                    timed_bindings.append((binding, time))
        # timed bindings are only enabled if they have time <= clock
        # if there are no such bindings, set the clock to the earliest time at which there are
        timed_bindings.sort(key=lambda b: b[1])
        if len(timed_bindings) > 0 and timed_bindings[0][1] > self.clock:
            self.clock = timed_bindings[0][1]
        # now return the untimed bindings + the timed bindings that have time <= clock
        return untimed_bindings + [(binding, time) for (binding, time) in timed_bindings if time <= self.clock]

    def fire(self, timed_binding):
        """
        Fires the specified timed binding.
        Binding is a tuple ([(arc, token), (arc, token), ...], time)
        """
        (binding, time) = timed_binding
        # process incoming places:
        transition = None
        variable_assignment = dict()
        for (arc, token) in binding:
            transition = arc.dst
            # remove tokens from incoming places
            arc.src.remove_token(token)
            # assign values to the variables on the arcs
            if arc.inscription is not None:
                variable_assignment[arc.inscription] = token.color

        # process outgoing places:
        for arc in transition.outgoing:
            # add tokens on outgoing places
            token = Token(None)
            if arc.inscription is not None:  # if the arc has an inscription
                exec(compile(self.additional_functions + "result = " + arc.inscription, "<string>", "exec"), variable_assignment)
                token.color = variable_assignment['result']
            if arc.time_expression is not None:  # if the transition has a time expression
                exec(compile(self.additional_functions + "result = " + str(arc.time_expression), "<string>", "exec"), variable_assignment)
                token.time = self.clock + variable_assignment['result']
            arc.dst.add_token(token)

        print(f"Fired transition with binding {binding}")

    def simulation_run(self, length):
        run = []
        i = 0
        active_model = True
        while i < length and active_model:
            bindings = self.bindings()
            if len(bindings) > 0:
                binding = random.choice(bindings)
                run.append(binding)
                self.fire(binding)
                i += 1
            else:
                active_model = False
        return run
        

class AEPetriNet(PetriNet):
    """
    Action-Evolution Petri Net with internal clock. Transitions are timed and have a tag (a or e).
    Only transtions with the same tag as the network tag can fire.
    Timed transitions exhibit a time window (tw) for tokens to fire
    that is to be intended as (self.clock + transition.tw_low, self.clock + transition.tw_high).
    Only timed tokens with time within the time window can be used to fire the transition.
    """
    def __init__(self, additional_functions, tag='e', observation_type='vector', length=1000):
        super().__init__(additional_functions)
        self.actions_dict = None
        self.rewards = 0
        self.tag = tag
        self.observation_type = observation_type
        self.length = length #length of the simulation run

        if observation_type == 'vector': #if observation type is 'graph', color_set is not used
            self.colors_set = None #if colors sets are not defined on single places, only initial colors are used for bulding observations
    
    def sort_attributes(self):
        self.places.sort(key=lambda p: p._id)
        self.transitions.sort(key=lambda t: t._id)
        self.arcs.sort(key=lambda t: t._id)

    def bindings(self):
        """
        Calculates the set of timed bindings that is enabled in the net.
        Each binding is a tuple ([(arc, token), (arc, token), ...], time) that represents a single enabling binding.
        If no timed binding is enabled at the current clock time, updates the current clock time to the earliest time at which there is.
        :return: list of tuples ([(arc, token), (arc, token), ...], time)
        """
        active_model = True
        untimed_bindings = []
        timed_bindings = []
        for t in [t for t in self.transitions]:
            for (binding, time) in self.transition_bindings(t):
                if time is None:
                    untimed_bindings.append((binding, time))
                else:
                    timed_bindings.append((binding, time, t))
        # timed bindings are only enabled if they have time <= clock
        # if there are no such bindings, set the clock to the earliest time at which there are
        timed_bindings_curr = [(binding, time) for (binding, time, t) in timed_bindings if t.tag == self.tag] #timed bindings for the current tag
        timed_bindings_other = [(binding, time) for (binding, time, t) in timed_bindings if t.tag != self.tag] #timed bindings for the other tag

        timed_bindings.sort(key=lambda b: b[1])
        timed_bindings_curr.sort(key=lambda b: b[1])
        timed_bindings_other.sort(key=lambda b: b[1])

        #then check if the tag has to be updated (TODO: simplify this condition)
        if (len(timed_bindings_curr) > 0 and timed_bindings_curr[0][1] <= self.clock):
            pass
        elif (len(timed_bindings_curr) > 0 and timed_bindings_curr[0][1] > self.clock):
            if len(timed_bindings_other) > 0:
                if (timed_bindings_other[0][1] < timed_bindings_curr[0][1]):
                    self.switch_tag()
                    self.clock = timed_bindings_other[0][1]
                    print(f"Tag switched to {self.tag} at time {self.clock}")
                    timed_bindings_curr = timed_bindings_other
                else:
                    self.clock = timed_bindings_curr[0][1]
            else:
                self.clock = timed_bindings_curr[0][1]
        elif len(timed_bindings_curr) == 0 and len(timed_bindings_other) != 0:
            self.switch_tag()
            self.clock = timed_bindings_other[0][1]
            print(f"Tag switched to {self.tag} at time {self.clock}")
            timed_bindings_curr = timed_bindings_other
        else:
            active_model = False

        # now return the untimed bindings + the timed bindings that have time <= clock
        bindings = untimed_bindings + [(binding, time) for (binding, time) in timed_bindings_curr if time <= self.clock]
        return bindings, active_model

    def fire(self, timed_binding):
        """
        Fires the specified timed binding.
        Binding is a tuple ([(arc, token), (arc, token), ...], time)
        """
        (binding, time) = timed_binding
        # process incoming places:
        transition = None
        variable_assignment = dict()

        for (arc, token) in binding:
            transition = arc.dst
            # remove tokens from incoming places
            arc.src.remove_token(token)
            # assign values to the variables on the arcs
            if arc.inscription is not None:
                variable_assignment[arc.inscription] = token.color

        # process outgoing places:
        for arc in transition.outgoing:
            # add tokens on outgoing places
            token = Token(None)
            if arc.inscription is not None:  # if the arc has an inscription
                #import pdb; pdb.set_trace()
                exec(compile(self.additional_functions + "result = " + arc.inscription, "<string>", "exec"), variable_assignment)
                token.color = variable_assignment['result']
            if arc.time_expression is not None:  # if the transition has a time expression
                exec(compile(self.additional_functions + "result = " + str(arc.time_expression), "<string>", "exec"), variable_assignment)
                token.time = self.clock + variable_assignment['result']
            arc.dst.add_token(token)

        self.update_reward(timed_binding)

        print(f"Fired transition with binding {binding}")

    def update_reward(self, timed_binding):
        (binding, time) = timed_binding
        transition = binding[0][0].dst #transition is always the same
        if transition.reward != None:
            if type(transition.reward) == int or type(transition.reward) == float:
               self.rewards += transition.reward
            elif type(transition.reward) == str: #rewards can be awarded on the base of a reward function
                
                # process incoming places:
                variable_assignment = dict()
                for (arc, token) in binding:
                    # assign values to the variables on the arcs
                    if arc.inscription is not None:
                        variable_assignment[arc.inscription] = token.color
                exec(compile(self.additional_functions + "result = " + transition.reward, "<string>", "exec"), variable_assignment)
                self.rewards += variable_assignment['result']


    def switch_tag(self):
        if self.tag == 'a':
            self.tag = 'e'
        elif self.tag == 'e':
            self.tag = 'a'

    def get_to_first_action(self):
        """
        Runs the petri net until it gets to 'a' tag (initialization for mdp environment)
        """
        bindings, active_model = self.bindings()

        while self.tag == 'e':
            bindings, active_model = self.bindings()
            if self.tag == 'a': #the bindings() function controls the evolution of self.tag, so we need to break the cicle as soon as tag == 'a'
                break
            if len(bindings) > 0:
                binding = random.choice(bindings)
                self.fire(binding)
            else:
                raise Exception("Invalid initial state for the network")
        return self

    def get_actions(self):
        """
        Provide the color sets of all places connected to arcs incoming to "a" tagged transitions. Necessary to define the actions space.
        If the color set is not defined on at least one place, the set of actions is given by the set of colors of all tokens in the network and a warning is raised
        """
        if self.observation_type == 'vector':
            if not all([p.colors_set for p in self.places]):
                # logging.warning("At least one color set was not specified! Using all tokens' colors for determining possible actions")

                # create a list of size n elements, each a tuple, where n is the amount of places connected for every 'a' transition in the network, for every color present in the node
                color_associations = []
                colors_set = list({t.color for t in [item for sublist in self.places for item in sublist.marking]})

                for t in [t for t in self.transitions if t.tag == 'a']:
                    n_inc_arcs = len([a for a in self.arcs if a.dst == t])
                    associations = list(itertools.product(colors_set,
                                                          repeat=n_inc_arcs))  # all combinations of size n_inc_arcs, possibly with repeating elements
                    variables = [a.inscription for a in self.arcs if
                                 a in t.incoming]  # all the variables of arcs incoming to t

                    # create list of dictionaries, each of which contains an association that is a possible binding
                    variable_values_list = []
                    for ass in associations:
                        variable_values_list.append(dict(zip(variables, ass)))

                    # evaluate guard for each fictious binding to determine possible associations
                    for variable_values in variable_values_list:
                        if t.guard:
                            v_v_original = {**variable_values}
                            exec(compile(self.additional_functions + "result = " + t.guard, "<string>", "exec"),
                                 variable_values)
                            enabled = variable_values['result']
                            if enabled: color_associations.append(v_v_original)
                        else:
                            color_associations.append(**variable_values)

                return color_associations
                # return sorted(color_associations, key=lambda d: d['name']) #WARNING: not sorted
            else:
                # create a list of size n elements, each a tuple, where n is the amount of places connected for every 'a' transition in the network, for every color present in the node
                color_associations = []

                for t in [t for t in self.transitions if t.tag == 'a']:
                    # get colors_set attribute of all places that are the src of arcs whose dst is a transition with tag=='a'
                    colors_set = [list(p.colors_set) for p in self.places if any([a.dst == t for a in p.outgoing])]

                    n_inc_arcs = len([a for a in self.arcs if a.dst == t])
                    # associations = list(itertools.product(*colors_set))#all combinations of size n_inc_arcs, possibly with repeating elements
                    associations = list(itertools.product(*colors_set))
                    variables = [a.inscription for a in self.arcs if
                                 a in t.incoming]  # all the variables of arcs incoming to t

                    # create list of dictionaries, each of which contains an association that is a possible binding
                    variable_values_list = []
                    for ass in associations:
                        variable_values_list.append(dict(zip(variables, ass)))

                    # evaluate guard for each fictious binding to determine possible associations
                    for variable_values in variable_values_list:
                        if t.guard:
                            v_v_original = {**variable_values}
                            exec(compile(self.additional_functions + "result = " + t.guard, "<string>", "exec"),
                                 variable_values)
                            enabled = variable_values['result']
                            if enabled: color_associations.append(v_v_original)
                        else:
                            color_associations.append(variable_values)

                color_associations_strings = sorted([json.dumps(a) for a in
                                                     color_associations])  # sort alphabetically to ensure the same positions in subsequent runs
                color_associations = [json.loads(a) for a in color_associations_strings]
                return color_associations
        elif self.observation_type == 'graph':
            """
            Return a list of all the available actions in the form of list of nodes indexes in the graph corresponding to
            the petri net
            """
            return self.actions_dict
        else:
            raise Exception("Invalid value for parameter obs_type")


    def get_valid_actions(self):
        """
        Valid actions are given by the colors of tokens in places connected to arcs incoming to "a" tagged transitions
        """
        if self.obs_type == 'vector':
            bindings = self.bindings()[0] #take all bindings without time
            inscr_color_bindings = self.bindings_to_associations(bindings)

            return inscr_color_bindings
        elif self.obs_type == 'graph':
            """
            Return a list of all the available actions in the form of list of nodes indexes in the graph corresponding to
            the petri net
            """
            return list(self.actions_dict.keys())

    def get_observation(self):
        """
        Return an observation in the form specified by self.observation_type
        """
        if self.observation_type == 'vector':
            #vector observation expects places to contain their color sets
            if not all([p.colors_set for p in self.places]) and not self.colors_set:
                #logging.warning("At least one color set was not specified! Using all tokens' colors for determining possible actions")
                self.colors_set = sorted({t.color for t in [item for sublist in self.places for item in sublist.marking]}) #initial colors set

            if self.colors_set: #if the color sets are globally defined, only initial colors are tracked
                ret_vec = []
                for p in self.places:
                    #logging.warning(f"A color set was not specified! Using all tokens' colors for building observation for place {p}")
                    for c in self.colors_set:
                        for p in self.places:
                            ret_vec.append(sum(1 for x in p.marking if x.color.split('@')[0] == c)) #count occurrences of tokens with given color in the place's marking (assumes timed tokens)
            else: #colors sets are defined on each place: the amount of tokens of each type in the color set is tracked
                ret_vec = []
                for p in self.places:
                    #logging.warning(f"A color set was not specified! Using all tokens' colors for building observation for place {p}")
                    for c in p.colors_set:
                        for p in self.places:
                            ret_vec.append(sum(1 for x in p.marking if x.color.split('@')[0] == c)) #count occurrences of tokens with given color in the place's marking (assumes timed tokens)
            ret = ret_vec
        elif self.observation_type == 'graph':
            #graph observation expects places to contain their color sets types (only the types of variables for each token)
            ret_graph = HeteroData()
            mask_graph = HeteroData()
            pn_bindings = {} #the observation contains the bindings corresponding to expanded action nodes
            node_types = [p._id for p in self.places]#[set(p.tokens_attributes.keys()) for p in self.places]
            node_types.append('transition')

            self.expanded_pn = self.expand() #expand the petri net into a new petri net with 1-bounded places
            print(self.expanded_pn)

            # Update the HeteroData object with the places attributes and the transition nodes
            for n_t in node_types:
                if n_t != 'transition':
                    p_nodes = []
                    for p in self.expanded_pn.places:
                        if p._id.split('.')[0] == n_t:
                            for token in p.marking:  # will always be only one
                                token_value = torch.tensor([value for key, value in token.color.items()]).type(torch.float32)
                                p_nodes.append(token_value)
                    if p_nodes:
                        ret_graph[n_t].x = torch.stack(p_nodes)
                    else:
                        #import pdb; pdb.set_trace()
                        ret_graph[n_t].x = torch.empty(0, len(self.by_id[n_t].tokens_attributes.keys())) #create empty placeholder of the right size
                        print(f"Warning: no tokens of type {n_t} found in the marking. Skipping to the next node type")
                else:
                    t_nodes = []
                    for i, t in enumerate(self.expanded_pn.transitions):
                        if t.tag == 'e':
                            t_nodes.append(torch.tensor([0]).type(torch.float32))
                        elif t.tag == 'a':
                            incoming = [(self.by_id[f'({a.src._id.split(".")[0]}, {a.dst._id.split(".")[0]})'] , a.src) for a in self.expanded_pn.arcs if a.dst._id == t._id]
                            b_list = [(el[0], el[1].marking[0]) for el in incoming]
                            b_time = min([t[1].time for t in b_list])
                            pn_bindings[i] = (b_list, b_time)
                            t_nodes.append(torch.tensor([1]).type(torch.float32))

                    ret_graph[n_t].x = torch.stack(t_nodes)
                    mask_graph[n_t].x = torch.stack(t_nodes)

            #import pdb; pdb.set_trace()
            # Update the HeteroData object with the arcs
            edge_types = ['edge'] # possibly expand with something like [a.inscription for a in expanded_pn.arcs]
            for e_t in edge_types:
                if e_t != 'self_loop': #currently self loops are not supported
                    for a in self.expanded_pn.arcs:
                        if ret_graph[a.src._id.split('.')[0]]: #if the place has tokens inside of it
                            #ret_graph[a.src._id.split('.')[0], 'edge', a.dst._id.split('.')[0]].edge_index = torch.tensor([[i for i in range(len(ret_graph[a.src._id.split('.')[0]].x))], [i for i in range(len(ret_graph[a.src._id.split('.')[0]].x))]]).type(torch.float32)
                            ret_graph[
                                a.src._id.split('.')[0], 'edge', 'transition'].edge_index = torch.tensor(
                                [[i for i in range(len(ret_graph[a.src._id.split('.')[0]].x))],
                                 [i for i in range(len(ret_graph[a.src._id.split('.')[0]].x))]]).type(torch.int64)
                        if ret_graph[a.dst._id.split('.')[0]]: #if the place has tokens inside of it
                            #ret_graph[a.src._id.split('.')[0], 'edge', a.dst._id.split('.')[0]].edge_index = torch.tensor([[i for i in range(len(ret_graph[a.dst._id.split('.')[0]].x))], [i for i in range(len(ret_graph[a.dst._id.split('.')[0]].x))]]).type(torch.float32)
                            ret_graph[
                                'transition', 'edge', a.dst._id.split('.')[0]].edge_index = torch.tensor(
                                [[i for i in range(len(ret_graph[a.dst._id.split('.')[0]].x))],
                                 [i for i in range(len(ret_graph[a.dst._id.split('.')[0]].x))]]).type(torch.int64)
                    for place in self.places: #create empyty placeholders for arcs that are not represented in the expanded network
                        #import pdb;
                        #pdb.set_trace()
                        if not ret_graph['transition', 'edge', place._id]:
                            ret_graph['transition', 'edge', place._id].edge_index = torch.empty(2, 0).type(torch.int64)
                            mask_graph['transition', 'edge', place._id].edge_index = torch.empty(2, 0).type(torch.int64)
                        if not ret_graph[place._id, 'edge', 'transition']:
                            ret_graph[place._id, 'edge', 'transition'].edge_index = torch.empty(2, 0).type(torch.int64)
                            mask_graph[place._id, 'edge', 'transition'].edge_index = torch.empty(2, 0).type(torch.int64)

                    #import pdb; pdb.set_trace()
                    print("Added missing edges")


            self.actions_dict = pn_bindings
            return {'graph': ret_graph, 'mask': mask_graph, 'actions_dict': pn_bindings}

        else:
            raise Exception("Invalid value for parameter obs_type")

        return ret

    def expand(self):
        """
        A-E PN expansion is a function from attributed A-E PN A to (1-bounded) attributed A-E PN B that maps every token
        in A to a place in B and every transition from places in A to one or more transitions in B so that if a place in
        A is connected to a transition in A (or vice-versa), for each token in that place there is a transition from the
        mapped token in B to a new transition in B.
        """
        expanded_pn = AEPetriNet(self.additional_functions, self.tag, self.observation_type)
        exp_pn_base_places = []
        for p in self.places:
            for i, t in enumerate(p.marking): #for each token in the place, create a 1-bounded place with the token inside
                expanded_pn.add_place(Place(p._id + '.' + str(i), tokens_attributes=p.tokens_attributes))
                expanded_pn.add_mark_by_id(p._id + '.' + str(i), t.color, time=t.time)
                exp_pn_base_places.append(p)

        exp_pn_base_places = set(exp_pn_base_places)

        for t in self.transitions:

            if t.tag == 'e': #E transitions are not expanded (they are not affected by the action) def __init__(self, _id, tag, reward, guard=None):
                expanded_pn.add_transition(TaggedTransition(t._id, tag=t.tag, reward=t.reward, guard=t.guard))
#                for i, b in enumerate(transition_bindings): #TODO: make sure this works in every situation
                for arc in t.incoming:
                    for idx in range(len(arc.src.marking)):
                        expanded_pn.add_arc_by_ids(arc.src._id + '.' + str(idx), arc.dst._id, arc.inscription,
                                                   arc.time_expression)

                # outgoing are the same for a and e transitions
                for arc in t.outgoing:
                    for idx in range(len(arc.dst.marking)):
                        expanded_pn.add_arc_by_ids(arc.src._id, arc.dst._id + '.' + str(idx), arc.inscription,
                                                   arc.time_expression)

            elif t.tag == 'a': #A transitions are expanded into n transitions where n is the number of possible tokens' associations from connected places
                transition_bindings = self.transition_bindings(t)  # operated on the non-expanded network
                #import pdb; pdb.set_trace()
                for i, b in enumerate(transition_bindings):
                    expanded_pn.add_transition(TaggedTransition(t._id + '.' + str(i), tag=t.tag, reward=t.reward, guard=t.guard))
                    for arc, token in b[0]:
                        for place in expanded_pn.places:
                            if place._id.split('.')[0] == arc.src._id and place.marking[0] == token:
                                expanded_pn.add_arc_by_ids(place._id, arc.dst._id + '.' + str(i), arc.inscription, arc.time_expression)
                    for arc in self.arcs:
                        if arc.src == t:
                            for place in expanded_pn.places:
                                if place._id.split('.')[0] == arc.dst._id:
                                    expanded_pn.add_arc_by_ids(t._id + '.' + str(i), place._id, arc.inscription, arc.time_expression)
                    else:
                        pass

            else:
                raise Exception("Invalid tag for transition")


        #import pdb; pdb.set_trace()
        return expanded_pn


    def bindings_to_associations(self, bindings):
        #it should work seamlessly regardless of the observation type
        inscr_color_bindings = []

        for b in bindings:
            unt_binding = b[0]
            dict_res = {}
            for u_b in unt_binding:
                dict_res[u_b[0].inscription] = u_b[1].color
            inscr_color_bindings.append(dict_res)

        return inscr_color_bindings

    def apply_action(self, action):
        """
        Applies the action chosen by the RL algorithm to the network
        """
        bindings, active_model = self.bindings()

        if self.observation_type == 'vector':
            inscr_color_bindings = self.bindings_to_associations(bindings)

            action_bindings = [bindings[ind] for ind, b in enumerate(inscr_color_bindings) if b==action] #select only the bindings that comply with the chosen action

            #import pdb; pdb.set_trace();
            if len(action_bindings) > 0:
                prev_tag = self.tag
                binding = random.choice(action_bindings)
                self.fire(binding)
                self.bindings() #used to update the clock if necessary
                return self.get_observation(), 1
            else:
                raise Exception("The chosen action seems to be unavailable! No action will be performed.")

        elif self.observation_type == 'graph':
            #import pdb; pdb.set_trace()
            self.fire(action)
            self.bindings()
            return self.get_observation(), 1

    def get_nodes_edges_types(self):
        """
        Returns the types of nodes and edges in the graph representation of the petri net
        """
        nodes_list = [p._id for p in self.places]
        nodes_list.append('transition')
        edges_list = list(set([(arc.src._id, 'edge', arc.dst._id) for arc in self.arcs])) # possibly expand with something like [a.inscription for a in expanded_pn.arcs]
        return nodes_list, edges_list


    def simulation_run(self, length):
        run = []
        i = 0
        active_model = True
        prev_tag = self.tag
        while self.clock < length and active_model:
            bindings, active_model = self.bindings()
            if len(bindings) > 0:
                binding = random.choice(bindings)
                run.append(binding)
                self.fire(binding)
                print(binding)
                i += 1

        print(f"Total reward random over {length} steps: {self.rewards} \n")
        return run, self.rewards

    def training_run(self, num_steps, episode_length, load_model, out_model_name="model.pt", out_folder_name = './out'):
        """
        Runs the petri net as a reinforcement learning environment
        """


        env = new_aepn_env.AEPN_Env(self, length = episode_length)

        out_path = os.path.join(out_folder_name, out_model_name)
        #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
        # tensorboard --logdir ./tmp/
        # then, in a browser page, access localhost:6006 to see the board
        log_dir = os.path.join(out_folder_name, 'log')

        if not os.path.exists(out_folder_name):
            os.makedirs(out_folder_name)
            os.makedirs(log_dir)
            
        if load_model:
            model = MaskablePPO.load(out_path, env, n_steps = 50)
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps = 50, verbose=1)
        
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
        model.learn(total_timesteps=num_steps)
        env.reset()
        model.save(out_path)

        print("Training over")

    def new_training_run(self, test=False):
        args = make_parser().parse_args()
        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if args.agent_seed is not None:
            np.random.seed(args.agent_seed)
            random.seed(args.agent_seed)
            torch.manual_seed(args.agent_seed)
            torch.cuda.manual_seed(args.agent_seed)
            # TODO: two more lines for cuda

        env = new_aepn_env.AEPN_Env(self)
        #nodes_list, edges_list = self.get_nodes_edges_types()
        metadata = self.get_observation()['graph'].metadata()

        agent = make_agent(args, metadata=metadata)
        # Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
        # tensorboard --logdir .
        # then, in a browser page, access localhost:6006 to see the board
        logdir = make_logdir(args)
        print("Saving run in", logdir)
        if not test:
            print("Training...")
            agent.train(env, episodes=args.episodes, epochs=args.epochs,
                        save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                        max_episode_length=args.max_episode_length, batch_size=args.batch_size)
        else:

            shutil.copy(args.policy_weights, os.path.join(logdir, 'run', "policy-500.h5"))
            with open(os.path.join(logdir, "results.csv"), 'w') as f:
                f.write("Return,Length\n")
            for _ in range(args.episodes):
                reward, length = agent.run_episode(env, max_episode_length=args.max_episode_length) #CAREFUL: still to be reshaped
                with open(os.path.join(logdir, "results.csv"), 'a') as f:
                    f.write(f"{reward},{length}\n")

    def testing_run(self, length, additional_functions = None, out_model_name="model.pt", out_folder_name = './out'):
        """
        Loads a trained model and performs length steps of evaluation
        """
        self.length = length

        out_path = os.path.join('out', out_model_name)
        model = MaskablePPO.load(out_path)
        env = aepn_env.AEPN_Env(self)
        env.reset()

        
        done = False
        t_rewards = 0
        while not done:
            # Retrieve current action mask
            obs = self.get_observation()
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, rewards, done, info = env.step(action)
            t_rewards += rewards #unused


        total_reward_ppo = env.pn.rewards        
        return total_reward_ppo

    def new_testing_run(self, length, additional_functions=None, out_model_name="network-1.pth", out_folder_name='./out'):
        """
        Loads a trained model and performs length steps of evaluation
        """
        self.length = length

        env = new_aepn_env.AEPN_Env(self)
        #metadata = self.get_observation()['graph'].metadata()
        #metadata = env.reset()['graph'].metadata()
        #import pdb; pdb.set_trace()
        out_path = os.path.join('C:', os.sep, 'Users', '20215143', 'tue_repos', 'cpn_flex', 'cpn-project', 'cpn', 'data', 'train', 'run', out_model_name)#os.path.join('out', out_model_name)
        #load state dict
        model = torch.load(out_path)#HeteroActor(1, metadata)
        #model.reset_parameters()
        # Load the parameters
        #state_dict = torch.load(out_path)
        # Set the model to evaluation mode
        #model.load_state_dict(state_dict)
        #model.eval()



        done = False
        t_rewards = 0
        while not done:
            # Retrieve current action mask
            obs = self.get_observation()
            #action_masks = env.action_masks()
            action_probs = model(obs)
            action = int(torch.argmax(action_probs)) #get the maximum probability action
            obs, rewards, done, truncated, info = env.step(action)
            t_rewards += rewards  # unused

        total_reward_ppo = env.pn.rewards
        return total_reward_ppo

    def run_evolutions(self, run, i, active_model):
        """
        Function invoked by Gym environment to let the network perform evolutions when no actions are required
        """

        while self.clock <= self.length and active_model:
            bindings, active_model = self.bindings()
            if self.tag == 'a': #the bindings() function controls the evolution of self.tag, so we need to break the cicle as soon as tag == 'a'
                break
            if len(bindings) > 0 and self.tag == 'e':
                binding = random.choice(bindings)
                run.append(binding)
                self.fire(binding)
                #print(binding)
                i += 1
            elif len(bindings) > 0 and self.tag == 'a': #give control to the gym env by returning the current observation
                return self.get_observation(), self.clock > self.length, i
            else:
                active_model = False

        return self.get_observation(), self.clock > self.length or not active_model, i





if __name__ == "__main__":
    test_cpn = False #test standard cpn implementation
    
    test_task_assignment = True #test a-e cpn for task assignment problem
    
    test_simulation = False
    test_training = True#True
    test_inference = False

    f = open('./gym_env/additional_functions/new_color_functions.py', 'r')
    temp = f.read()
    f.close()
    my_functions = temp
    pn = AEPetriNet(my_functions, observation_type='graph')

    if test_task_assignment:

        pn.add_place(Place("arrival", tokens_attributes={"id": "int", "type": "int"})) #attribute TIME is always added by default
        pn.add_place(Place("waiting", tokens_attributes={"id": "int", "type": "int"}))
        pn.add_place(Place("resources", tokens_attributes={"id": "int", "compatibility_0": "int", "compatibility_1": "int", "is_set": "int"}))
        pn.add_place(Place("busy", tokens_attributes={'resource_id': 'int', 'task_id': 'int', 'task_type': 'int', 'compatibility_0': 'int', 'compatibility_1': 'int'}))

        pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
        pn.add_transition(TaggedTransition("set_compatibility", guard="is_not_set(res)", tag='e', reward=0))
        pn.add_transition(TaggedTransition("start", tag='a', reward=0))
        pn.add_transition(TaggedTransition("complete", tag='e', reward=1))

        pn.add_arc_by_ids("arrival", "arrive", "x")
        pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
        pn.add_arc_by_ids("arrive", "waiting", "x", time_expression=0)

        pn.add_arc_by_ids("resources", "set_compatibility", "res")
        pn.add_arc_by_ids("set_compatibility", "resources", "set_compatibility(res)")

        pn.add_arc_by_ids("resources", "start", "resource")
        pn.add_arc_by_ids("waiting", "start", "case")
        pn.add_arc_by_ids("start", "busy", "new_combine(case, resource)", time_expression=1)

        pn.add_arc_by_ids("busy", "complete", "case_resource")
        pn.add_arc_by_ids("complete", "resources", "new_split(case_resource)", time_expression=0)

        pn.add_mark_by_id("resources", {'id': 0, 'compatibility_0': 0, 'compatibility_1': 1, 'is_set': 0}, 0)
        pn.add_mark_by_id("resources", {'id': 1, 'compatibility_0': 1, 'compatibility_1': 0, 'is_set': 0}, 0)
        pn.add_mark_by_id("arrival", {'id': 0, 'type': 0}, 0)
        pn.add_mark_by_id("arrival", {'id': 1, 'type': 1}, 0)

    print(pn)

    if test_simulation:
        test_run = pn.simulation_run(1000)
        for test_binding in test_run:
            print(test_binding)

    elif test_training:
        start_time = time.time()
        pn.new_training_run(test_inference)
        print("TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))

    elif test_inference:

        length = 100
        
        repetitions = 1000

        r_vec_ppo = []
        r_vec_random = []
        for i in range(repetitions):
            pn = copy.copy(pn)
            rand_pn = copy.deepcopy(pn)

            total_reward_ppo = pn.new_testing_run(length, additional_functions = my_functions)
            r_vec_ppo.append(total_reward_ppo)
            run, total_reward_random = rand_pn.simulation_run(length)
            r_vec_random.append(total_reward_random)
        import numpy as np
        print(f"Average reward PPO: {np.mean(r_vec_ppo)} with standard deviation {np.std(r_vec_ppo)}\nAverage reward random: {np.mean(r_vec_random)} with standard deviation {np.std(r_vec_random)}")