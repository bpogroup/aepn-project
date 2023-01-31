import random
import json
import itertools
from tkinter import SEL
import uuid
from abc import ABC, abstractclassmethod
from pncomponents import Transition, Place, Token, Arc, TimeWindowedArc, TimeIncreasingArc, TimedToken, Color
import os
from gym_env import aepn_env
import logging
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

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
    def __init__(self):
        self.places = []
        self.transitions = []
        self.arcs = []
        
        self.by_id = dict()

    def __str__(self):
        result = ""
        result += "P={"
        for i in range(len(self.places)):
            result += str(self.places[i]) + "()"
            if i < len(self.places) - 1:
                result += ","
        result += "}\n"
        result += "T={"
        for i in range(len(self.transitions)):
            result += str(self.transitions[i]) + "()"
            if i < len(self.transitions) - 1:
                result += ","
        result += "}\n"
        result += "A={"
        for i in range(len(self.arcs)):
            result += "(" + str(self.arcs[i].src) + "," + str(self.arcs[i].dst) + "," + str(self.arcs[i].var) + ")"
            if i < len(self.arcs) - 1:
                result += ","
        result += "}\n"        
        result += "M={"
        for p in self.places:
            result += "(" + str(p) + ":" 
            for k, v in p.print_marking().items():
                result += "(" + str(k) + ", " + str(v) + ")\n"
            result += ")"
        result += "}"                
        return result

    def add_place(self, place):
        self.places.append(place)
        self.by_id[place._id] = place

    def add_transition(self, transition):
        self.transitions.append(transition)
        self.by_id[transition._id] = transition

    def add_arc(self, src, dst, var=None, tw=None, increment=None, increment_type=None, increment_params=None):
        """
        Adds an arc to the Petri net from a source to a destination.
        An arc can have a variable by which a token value will be taken from the arcs incoming place (if any)
        or placed on the arcs outgoing place (if any).
        The variable can also be None (the default), in which case we assume the arc only processes 'black' tokens.
        :param src: a place or transition
        :param dst: a place or transition
        :param var: a string or None if there is no variable.
        """
        if not src in self.places and not src in self.transitions:
            raise Exception("Attempting to create an arc from a non-existing source.")
        if not dst in self.places and not dst in self.transitions:
            raise Exception("Attempting to create an arc to a non-existing destination.")
        if (src in self.places and dst in self.places):
            raise Exception("Attempting to create an arc from a place to a place.")
        if (src in self.transitions and dst in self.transitions):
            raise Exception("Attempting to create an arc from a transition to a transition.")
        
        if tw is not None and tw != (None, None):
            arc = TimeWindowedArc(src, dst, var, tw)
        elif increment is not None:
            arc = TimeIncreasingArc(src, dst, var, increment=increment)
        elif increment_type is not None and increment_params is not None:
            arc = TimeIncreasingArc(src, dst, var, delay_type = increment_type, params = increment_params) #randomized increment
        else:
            arc = Arc(src, dst, var)
        self.arcs.append(arc)

        #TODO: there is probably a more elegant way to do this
        for n in self.places:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break
        for n in self.transitions:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break

    def add_arc_by_ids(self, src, dst, var=None, tw=None, increment=None, increment_type=None, increment_params=None):
        if not src in self.by_id:
            raise Exception("Cannot find node with id '" + src + "' to construct arc to '" + dst + "'.")
        if not dst in self.by_id:
            raise Exception("Cannot find node with id '" + dst + "' to construct arc from '" + src + "'.")
        self.add_arc(self.by_id[src], self.by_id[dst], var, tw, increment, increment_type, increment_params)

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

    def add_mark_by_id(self, place, tokens_num):
        """
        :param place: a place
        :param tokens: a dictionary value -> count
        """

        if not place in self.by_id:
            raise Exception("Cannot find place with id '" + place + "' to mark.")
        for (t, n) in tokens_num.items():
            for i in range(n):
                self.add_mark(self.by_id[place], Token(_id=uuid.uuid1(), color=Color(json.loads(t)))) #adds a token with random id and given color to the chosen place

    def tokens_combinations(self, transition):
        # create all possible combinations of incoming token values
        bindings = [[]]
        for arc in transition.incoming:
            new_bindings = []
            for token_value in set([t.color for t in arc.src.marking]): #get set of colors in incoming place
                for binding in bindings:
                    new_binding = binding.copy()
                    new_binding.append((arc, (token_value, sum(map(lambda x : (x.color == token_value), arc.src.marking)))))
                    new_bindings.append(new_binding)
            bindings = new_bindings
        return bindings

    def transition_bindings(self, transition):
        """
        Calculates the set of bindings that enables the given transition.
        Each binding is a list of lists of (arc, token value) pairs, in which
        each list of pairs represents a single enabling binding.
        :param transition: the transition for which to calculate the enabling bindings.
        :return: a list of lists of (arc, token value) pairs
        """
        if len(transition.incoming) == 0:
            raise Exception("Though it is strictly speaking possible, we do not allow transitions like '" + str(self) + "' without incoming arcs.")

        bindings = self.tokens_combinations(transition)

        #transition.evaluate_guard_func() TODO!!

        # a binding must have all incoming arcs
        nr_incoming_arcs = len(transition.incoming)
        new_bindings = []
        for binding in bindings:
            if len(binding) == nr_incoming_arcs:
                new_bindings.append(binding)
        bindings = new_bindings

        #OLD: if two arcs have matching arc variables they should also have matching values for a binding to be enabling
        #NEW: incoming arcs should have matching values for a binding to be enabled
        variable_values = dict()
        result = []
        for binding in bindings:
            enabled = True
            if not self.color_is_present(binding):
                enabled = False
                break

            colors = [b[1][0] for b in binding]
            for i in range(len(colors)-1):
                if not colors[i].check_compatibility(colors[i+1]):
                    enabled = False
                    break
            if enabled:
                result.append(binding)
        return result

    def color_is_present(self, binding):
        for b in binding:
            colors_in_source = [t.color.color_list() for t in b[0].src.marking]
            if b[1][0].color_list() not in colors_in_source:
                return False
        return True

    def bindings(self):
        """
        Calculates the set of bindings that is enabled in the net.
        Each binding is a list of lists of (arc, token value) pairs, in which
        each list of pairs represents a single enabling binding.
        :return: a list of lists of (arc, token value) pairs
        """
        result = []
        for t in self.transitions:
            result += self.transition_bindings(t)
        return result

    def fire(self, binding):
        """
        Fires the specified binding.
        Changes the marking of the Petri net accordingly.
        """
        # process incoming places:
        transition = None
        variable_assignment = dict()
        for (arc, token_value) in binding:
            transition = arc.dst
            # remove tokens from incoming places
            incoming_place = arc.src
            token = random.choice([t for t in incoming_place.marking if set(t.color) & set(json.loads(token_value[0]))]) #choose a random token with given color
            incoming_place.remove_token(token) #remove the chosen token from the incoming place marking
            if arc.var is not None and token.color is not None: #cpn requires arc inscriptions
                variable_assignment[arc.var] = token.color
            elif token.color is None: #non colored pn assign a new token to the output (it creates a new one with new id)
                variable_assignment[arc.var] = None
            else:
                raise Exception("Invalid variable assignment detected. Check consistency of tokens and arcs.")

        # process outgoing places:
        for arc in [a for a in self.arcs if a.src==transition]: #self.__outgoing[transition]:
            arc.dst.add_token(Token(_id=uuid.uuid1(), color=variable_assignment[arc.var]))

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
    def __init__(self):
        super().__init__()
        self.clock = 0
        self.rewards = 0
        self.tag = 'e'
        #self.evolution_params = {(['a'], ['a']) : 0.7, (['b'], ['b']) : 0.7, (['a', 'b'], ['a']) : 0.2, (['a'], ['a', 'b']) : 0.2, (['a', 'b'], ['b']) : 0.2, (['b'], ['a', 'b']) : 0.2}

    def add_mark_by_id(self, place, tokens_num):
        """
        :param place: a place
        :param tokens: a dictionary value -> count
        """

        if not place in self.by_id:
            raise Exception("Cannot find place with id '" + place + "' to mark.")
        for (t, n) in tokens_num.items():
            for i in range(n):
                t = json.loads(t)
                if len(t) == 1:
                    self.add_mark(self.by_id[place], TimedToken(_id=uuid.uuid1(), color = Color(t), time=0)) #tokens are assumed to have time 0 at their creation (can be changed)
                else:
                    raise Exception("Token structure is invalid (tokens with multiple colors are currently not modeled)")

    def tokens_combinations(self, transition):
        # create all possible combinations of incoming token values (when token.time is lower than self.clock)
        bindings = [[]]
        for arc in transition.incoming:
            new_bindings = []
            for token_value in {x.color for x in arc.src.marking if x.time <= self.clock}: #get set of colors in incoming place (only for tokens in with age less than clock)
                for binding in bindings:
                    new_binding = binding.copy()
                    new_binding.append((arc, (token_value, sum(map(lambda x : (x.color.check_compatibility(token_value)), arc.src.marking)))))
                    new_bindings.append(new_binding)
            bindings = new_bindings
        return bindings


    def bindings(self):
        """
        Calculates the set of bindings that is enabled in the net.
        Each binding is a list of lists of (arc, token value) pairs, in which
        each list of pairs represents a single enabling binding.
        :return: a list of lists of (arc, token value) pairs
        """
        result = []
        for t in [t for t in self.transitions if t.tag == self.tag]:
            result += self.transition_bindings(t)

        if result == []:
            self.update_clock()
        
        return result

    def fire(self, binding):
        """
        Fires the specified binding.
        Changes the marking of the Petri net accordingly.
        """
        # process incoming places:
        transition = None
        variable_assignment = dict()
        for (arc, token_value) in binding:
            #TODO: extend with arc inscriptions (which tokens goes where) and probabilities
            transition = arc.dst
            # remove tokens from incoming places
            incoming_place = arc.src
            

            if len(arc.var.split(';')) == 1 and token_value is not None: 
                token = random.choice([t for t in incoming_place.marking if self.is_sublist(list(token_value[0].color_dict.values()), list(t.color.color_dict.values())) and t.time <= self.clock]) #choose a random token with given color
                incoming_place.remove_token(token) #remove the chosen token from the incoming place marking
                variable_assignment[arc.var] = token.color.color_list()[0]
            elif len(arc.var.split(';')) > 1 and token_value is not None: #more than one incoming arc
                token = random.choice([t for t in incoming_place.marking if self.is_sublist(list(token_value[0].color_dict.values()), list(t.color.color_dict.values())) and t.time <= self.clock]) #choose a random token with given color
                incoming_place.remove_token(token) #remove the chosen token from the incoming place marking
                for i in arc.var.split(';'):
                    variable_assignment[i] = list(token.color.color_dict[i])
            elif token_value is None: #non colored pn assign a new token to the output (it creates a new one with new id)
                token = random.choice([t for t in incoming_place.marking if (set(json.loads(token_value[0])) & set(json.loads(t.color))) and t.time <= self.clock]) #choose a random token with given color
                incoming_place.remove_token(token) #remove the chosen token from the incoming place marking
                variable_assignment[arc.var] = None
            else:
                raise Exception("Invalid variable assignment detected. Check consistency of tokens and arcs.")

        # process outgoing places: if more than one transition contributed to the firing, the outgoing token has multiple colors 
        for arc in [a for a in self.arcs if a.src==transition]:
            if len(arc.var.split(';')) == 1: #in the case where more than one incoming arc is present, the resulting color is the union of incoming colors
                new_token = TimedToken(_id=uuid.uuid1(), color=Color({arc.var : variable_assignment[arc.var]}), time = 0)
            else: #if the arc variable contains more than one entry, we must merge the incoming tokens' colors accordingly
                #TODO: define tokens' colors merge
                new_color = {}
                for i in arc.var.split(';'):
                    new_color[i] = variable_assignment[i]
                new_token = TimedToken(_id=uuid.uuid1(), color=Color(new_color), time = 0)
            arc.increment_time(new_token, self.clock)
            arc.dst.add_token(new_token)
        #increment reward
        self.rewards += transition.reward
        print(f"Reward updated after firing transition '{transition}'. New reward: {self.rewards}")

    def is_sublist(self, sublist, list):
       return all(x in list for x in sublist)

    def update_clock(self):
        self.clock += 1
        if self.tag == 'a':
            self.tag = 'e'
        elif self.tag == 'e':
            self.tag = 'a'

    def get_to_first_action(self):
        """
        Runs the petri net until it gets to 'a' tag (initialization for mdp environment)
        """
        while self.tag == 'e':
            bindings = self.bindings()
            if len(bindings) > 0:
                prev_tag = self.tag
                binding = random.choice(bindings)
                self.fire(binding)
            elif prev_tag != self.tag: #in this case, a tag change just happened (do not increment i)
                print(f"Tag switch from {prev_tag} to {self.tag}")
                prev_tag = self.tag
            else:
                raise Exception("Invalid initial state for the network")
        return self


    def get_actions(self):
        """
        Provide the color sets of all places connected to arcs incoming to "a" tagged transitions. Necessary to define the actions space.
        If the color set is not defined on at least one place, the set of actions is given by the set of colors of all tokens in the network and a warning is raised
        """
        if not all([p.colors_set for p in self.places]):
            #logging.warning("At least one color set was not specified! Using all tokens' colors for determining possible actions")
            colors_set = {t.color for t in [item for sublist in self.places for item in sublist.marking]}
            colors_associations = [(x.color_list(), y.color_list()) for x in colors_set for y in colors_set if x.check_compatibility(y)]
            colors_associations = list(colors_associations for colors_associations,_ in itertools.groupby(colors_associations))
            colors_ass_set = []
            for elem in colors_associations:
                if elem not in colors_ass_set: #and (elem[1], elem[0]) not in colors_ass_set:
                    colors_ass_set.append(elem)
            return sorted(colors_ass_set)
        else:
            raise Exception("Colors sets' management is not implemented yet")
            return {} #TODO: check color sets of nodes incoming to "a" transitions

    def get_valid_actions(self):
        """
        Valid actions are given by the colors of tokens in places connected to arcs incoming to "a" tagged transitions
        """
        val_b = self.bindings()
        valid_associations = [(b[0][1][0].color_list(), b[1][1][0].color_list()) for b in val_b]
        
        return valid_associations

    def marking_to_colors(self, list):
        """
        Helper function to extract a list of colors from a given list of tokens
        """
        ret_list = []
        for el in list:
            ret_list.append(el.color)

        return ret_list

    def get_observation(self, obs_type = 'vector'):
        """
        Return an observation in the form specified by obs_type (currently, only vector is available) TODO: implement matrix and graph obs_type
        """
        #the vector form uses 
        if not all([p.colors_set for p in self.places]):
            #logging.warning("At least one color set was not specified! Using all tokens' colors for determining possible actions")
            colors_set = {t.color for t in [item for sublist in self.places for item in sublist.marking]}
            colors_set_unique = []
            for elem in colors_set:
                for single_color in elem.color_list():
                    if single_color not in colors_set_unique:
                        colors_set_unique.append(single_color) #TODO: check if there is a more elegant way
            colors_set_unique = sorted(colors_set_unique)


        if obs_type == 'vector':
            ret_vec = []
            for p in self.places:
                if p.colors_set:
                    raise Exception("Colors set not yet implemented")
                else:
                    #logging.warning(f"A color set was not specified! Using all tokens' colors for building observation for place {p}")
                    for c in colors_set_unique:
                        for p in self.places:
                            ret_vec.append(sum(1 for x in p.marking if x == Token(-1, Color({'counter' : c})))) #count occurrences of tokens with given color in the place's marking
        else:
            raise Exception("Invalid value for parameter obs_type")

        return ret_vec

    def apply_action(self, action):
        """
        Applies the action chosen by the RL algorithm to the network
        """
        bindings = self.bindings()
        action_bindings = [b for b in bindings if (b[0][1][0].color_list() == action[0] and b[1][1][0].color_list() == action[1]) or (b[0][1][0].color_list() == action[1] and b[1][1][0].color_list() == action[0])] #select only the bindings that comply with the chosen action
        if len(action_bindings) > 0:
            prev_tag = self.tag
            binding = random.choice(action_bindings)
            self.fire(binding)
            print(binding)
            self.bindings() #used to update the clock if necessary
            return self.get_observation(), 1
        else:
            raise Exception("The chosen action seems to be unavailable! No action will be performed.")


    def simulation_run(self, length):
        run = []
        i = 0
        active_model = True
        prev_tag = self.tag
        while self.clock < length and active_model:
            
            bindings = self.bindings()
            if len(bindings) > 0:
                prev_tag = self.tag
                binding = random.choice(bindings)
                run.append(binding)
                self.fire(binding)
                print(binding)
                i += 1
            elif prev_tag != self.tag: #in this case, a tag change just happened (do not increment i)
                print(f"Tag switch from {prev_tag} to {self.tag}")
                prev_tag = self.tag
            else:
                active_model = False
        print(f"Total reward random over {length} steps: {self.rewards} \n")
        return run, self.rewards

    def training_run(self, num_episodes, episode_length, load_model, out_model_name="model.pt", out_folder_name = './out'):
        """
        Runs the petri net as a reinforcement learning environment
        """
        self.length = num_episodes

        env = aepn_env.AEPN_Env(self)
        if not os.path.exists(out_folder_name):
            os.makedirs(out_folder_name)
            
        out_path = os.path.join('out', out_model_name)
        
        for i in range(num_episodes):
            if load_model:
                model = MaskablePPO.load(out_path, env, n_steps = 50)
            else:
                model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps = 50, verbose=1)

            model.learn(total_timesteps=int(episode_length))
            env.reset()

        print("Training over")

        model.save(out_path)


    def testing_run(self, length, out_model_name="model.pt", out_folder_name = './out'):
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
            t_rewards += rewards

        print(f"Total reward PPO over {self.length} steps: {env.pn.rewards} \n \n")
        return self.rewards
        #self.__init__()

        #self.simulation_run(length)

        #total_reward_random = self.rewards

        #print(f"Total reward PPO: {total_reward_ppo} \n Total reward random: {total_reward_random}")



    def run_evolutions(self, run, i, active_model, prev_tag):
        """
        Function invoked by Gym environment to let the network perform evolutions when no actions are required
        """
        while self.clock <  self.length and active_model:
            bindings = self.bindings()
            if len(bindings) > 0 and self.tag == 'e':
                prev_tag = self.tag
                binding = random.choice(bindings)
                run.append(binding)
                self.fire(binding)
                print(binding)
                i += 1
            elif len(bindings) > 0 and self.tag == 'a': #give control to the gym env by returning the current observation
                return self.get_observation(), self.clock > self.length, i
            elif prev_tag != self.tag: #in this case, a tag change just happened (we do not increment i)
                print(f"Tag switch from {prev_tag} to {self.tag}")
                prev_tag = self.tag
            else:
                active_model = False

        return self.get_observation(), self.clock >= self.length or not active_model, i