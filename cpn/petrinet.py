import random


class PetriNet:
    def __init__(self):
        self.places = []
        self.transitions = []
        self.arcs = []
        self.marking = dict()
        
        self.__incoming = dict()
        self.__outgoing = dict()
        
        self.__by_id = dict()

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
            result += "(" + str(self.arcs[i][0]) + "," + str(self.arcs[i][1]) + "," + str(self.arcs[i][2]) + ")"
            if i < len(self.arcs) - 1:
                result += ","
        result += "}\n"        
        result += "M={"
        marking_keys = list(self.marking.keys())
        for i in range(len(marking_keys)):
            result += "(" + str(marking_keys[i]) + "," + str(self.marking[marking_keys[i]]) + ")"
            if i < len(self.marking) - 1:
                result += ","
        result += "}"                
        return result

    def add_place(self, place):
        self.places.append(place)
        self.marking[place] = dict()
        self.__incoming[place] = []
        self.__outgoing[place] = []
        self.__by_id[place.id] = place

    def add_transition(self, transition):
        self.transitions.append(transition)
        self.__incoming[transition] = []
        self.__outgoing[transition] = []
        self.__by_id[transition.id] = transition

    def add_arc(self, src, dst, var=None):
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
        arc = (src, dst, var)
        self.arcs.append(arc)
        self.__incoming[dst].append(arc)
        self.__outgoing[src].append(arc)

    def add_arc_by_ids(self, src, dst, var=None):
        if not src in self.__by_id:
            raise Exception("Cannot find node with id '" + src + "' to construct arc to '" + dst + "'.")
        if not dst in self.__by_id:
            raise Exception("Cannot find node with id '" + dst + "' to construct arc from '" + src + "'.")
        self.add_arc(self.__by_id[src], self.__by_id[dst], var)

    def add_mark(self, place, tokens):
        """
        Marks a place with tokens.
        Each token has a value (can also be None to indicate a black token).
        Tokens are represented as a dictionary with token values as keys
        and the number of tokens with that token value as dict_values.
        :param place: a place
        :param tokens: a dictionary value -> count
        """
        if not place in self.places:
            raise Exception("Attempting to mark a non-existing place.")
        self.marking[place] = tokens

    def add_mark_by_id(self, place, tokens):
        if not place in self.__by_id:
            raise Exception("Cannot find place with id '" + place + "' to mark.")
        self.add_mark(self.__by_id[place], tokens)

    def transition_bindings(self, transition):
        """
        Calculates the set of bindings that enables the given transition.
        Each binding is a list of lists of (arc, token value) pairs, in which
        each list of pairs represents a single enabling binding.
        Token value can be None to represent a black token.
        :param transition: the transition for which to calculate the enabling bindings.
        :return: a list of lists of (arc, token value) pairs
        """
        if len(self.__incoming[transition]) == 0:
            raise Exception("Though it is strictly speaking possible, we do not allow transitions like '" + str(transition) + "' without incoming arcs.")

        # create all possible combinations of incoming token values
        bindings = [[]]
        for arc in self.__incoming[transition]:
            new_bindings = []
            for token_value in self.marking[arc[0]]:
                for binding in bindings:
                    new_binding = binding.copy()
                    new_binding.append((arc, token_value))
                    new_bindings.append(new_binding)
            bindings = new_bindings

        # a binding must have all incoming arcs
        nr_incoming_arcs = len(self.__incoming[transition])
        new_bindings = []
        for binding in bindings:
            if len(binding) == nr_incoming_arcs:
                new_bindings.append(binding)
        bindings = new_bindings

        # if two arcs have matching arc variables they should also have matching values for a binding to be enabling
        variable_values = dict()
        result = []
        for binding in bindings:
            enabled = True
            for (arc, token_value) in binding:
                if arc[2] in variable_values:
                    if variable_values[arc[2]] != token_value:
                        enabled = False
                        break
                variable_values[arc[2]] = token_value
            if enabled:
                result.append(binding)
        return result

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
            transition = arc[1]
            # remove tokens from incoming places
            incoming_place = arc[0]
            self.marking[incoming_place][token_value] -= 1
            if self.marking[incoming_place][token_value] == 0:
                del self.marking[incoming_place][token_value]
            # assign values to the variables on the arcs
            if arc[2] is not None:
                variable_assignment[arc[2]] = token_value

        # process outgoing places:
        for arc in self.__outgoing[transition]:
            # add tokens on outgoing places
            value = None
            if arc[2] is not None:  # if the arc carries a variable
                if arc[2] not in variable_assignment:
                    raise Exception("Variable " + arc.variable + "' has no value in the assignment on tansition '" + str(transition) + "'.")
                value = variable_assignment[arc[2]]  # the value if the value that that variable has
            if value not in self.marking[arc[1]]:
                self.marking[arc[1]][value] = 0
            self.marking[arc[1]][value] += 1

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
        

class Place:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id
    
    def __repr__(self):
        return self.__str__()


class Transition:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.__str__()
