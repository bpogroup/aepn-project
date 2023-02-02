import random


class Node:
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


class Place(Node):
    def __init__(self, _id):
        """
        A place has an identifier _id and a marking.
        The marking is represented as a dictionary token -> number of tokens (with that color and time).
        :param _id: the identifier of the place.
        """
        super().__init__(_id)
        self.marking = dict()

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

    def __init__(self, _id, guard=None, time_expression=None):
        super().__init__(_id)
        self.guard = guard
        self.time_expression = time_expression


class Arc:
    def __init__(self, src, dst, inscription):
        self.src = src
        self.dst = dst
        self.inscription = inscription

    def __str__(self):
        return f"({self.src}, {self.dst})"

    def __repr__(self):
        return f"({self.src}, {self.dst})"


class PetriNet:
    def __init__(self, additional_functions=""):
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
            if self.transitions[i].time_expression is not None:
                result += "@+" + self.transitions[i].time_expression
            if i < len(self.transitions) - 1:
                result += ","
        result += "}\n"
        result += "A={"
        for i in range(len(self.arcs)):
            result += "(" + str(self.arcs[i].src) + "," + str(self.arcs[i].dst) + "," + str(self.arcs[i].inscription) + ")"
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

    def add_arc(self, src, dst, inscription=None):
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
        
        arc = Arc(src, dst, inscription)
        self.arcs.append(arc)

        # TODO: there is probably a more elegant way to do this
        for n in self.places:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break
        for n in self.transitions:
            if n == arc.dst or n == arc.src:
                n.connect_arc(arc)
                break

    def add_arc_by_ids(self, src, dst, inscription=None):
        if src not in self.by_id:
            raise Exception("Cannot find node with id '" + src + "' to construct arc to '" + dst + "'.")
        if dst not in self.by_id:
            raise Exception("Cannot find node with id '" + dst + "' to construct arc from '" + src + "'.")
        self.add_arc(self.by_id[src], self.by_id[dst], inscription=inscription)

    def add_mark(self, place, token, count=1):
        """
        Marks a place with a token.
        Each token has a value (can also be None to indicate a black token).
        :param place: a place
        :param token: the token to be added
        :param count: the number of tokens with that color and time to be added (defaults to 1)
        """
        if place not in self.places:
            raise Exception("Attempting to mark a non-existing place.")
        place.add_token(token, count)

    def add_mark_by_id(self, place_id, token, count=1):
        """
        Marks a place with a token. The place is identified by its identifier.
        Each token has a value (can also be None to indicate a black token).
        :param place_id: a place identifier
        :param token: the token to be added
        :param count: the number of tokens with that color and time to be added (defaults to 1)
        """
        if place_id not in self.by_id:
            raise Exception("Cannot find place with id '" + place_id + "'.")
        self.add_mark(self.by_id[place_id], token, count)

    def add_marks_by_id(self, place_id, tokens):
        """
        Marks a place with multiple tokens. The place is identified by its identifier.
        Each token has a value (can also be None to indicate a black token).
        :param place_id: a place identifier
        :param token2: the list of tokens to be added
        """
        for token in tokens:
            self.add_mark_by_id(place_id, token)

    def tokens_combinations(self, transition):
        # create all possible combinations of incoming token values
        bindings = [[]]
        for arc in transition.incoming:
            new_bindings = []
            for token in arc.src.marking.keys():  # get tokens in incoming place
                for binding in bindings:
                    new_binding = binding.copy()
                    new_binding.append((arc, token))
                    new_bindings.append(new_binding)
            bindings = new_bindings
        return bindings

    def transition_bindings(self, transition):
        """
        Calculates the set of timed bindings that enables the given transition.
        Each binding is a tuple ([(arc, token), ...], time) that represents a binding that is enabled at the specified clock time.
        time == None if the binding is not time-restricted.
        :param transition: the transition for which to calculate the enabling bindings.
        :return: list of tuples ([(arc, token), (arc, token), ...], time)
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
            if transition.time_expression is not None:  # if the transition has a time expression
                exec(compile(self.additional_functions + "result = " + transition.time_expression, "<string>", "exec"), variable_assignment)
                token.time = self.clock + variable_assignment['result']
            arc.dst.add_token(token)

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


def test_simple():
    my_functions = \
'''
def f(test):
    return test
'''
    pn = PetriNet(my_functions)

    pn.add_place(Place("arrival"))
    pn.add_place(Place("waiting"))
    pn.add_place(Place("resources"))
    pn.add_place(Place("busy"))

    pn.add_transition(Transition("arrive", time_expression="1"))
    pn.add_transition(Transition("start", "case==resource", time_expression="1"))
    pn.add_transition(Transition("complete", time_expression="0"))

    pn.add_arc_by_ids("arrival", "arrive", "x")
    pn.add_arc_by_ids("arrive", "arrival", "f(x)")
    pn.add_arc_by_ids("arrive", "waiting", "x")
    pn.add_arc_by_ids("waiting", "start", "case")
    pn.add_arc_by_ids("resources", "start", "resource")
    pn.add_arc_by_ids("start", "busy", "(case, resource)")
    pn.add_arc_by_ids("busy", "complete", "case_resource")
    pn.add_arc_by_ids("complete", "resources", "case_resource[1]")

    pn.add_mark_by_id("resources", Token("r1", 0))
    pn.add_mark_by_id("resources", Token("r2", 0))
    pn.add_mark_by_id("arrival", Token("r1", 0))

    print(pn)
    print()

    test_run = pn.simulation_run(10)
    for test_binding in test_run:
        print(test_binding)

if __name__ == "__main__":
    my_functions = \
'''
import random

B1 = 0
B2 = 1
B3 = 2
B4 = 3
B5 = 4
P1 = 5
P2 = 6
T1 = 7
S1 = 8
S2 = 9
S3 = 10
S4 = 11
S5 = 12
S6 = 13
S7 = 14
S8 = 15
S9 = 16
TIME = 17
'''
    pn = PetriNet(my_functions)

    # INIT
    pn.add_place(Place("to plan"))
    pn.add_mark_by_id("to plan", Token((0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0)))

    # PLAN
    pn.add_transition(Transition("plan", "b1 + b2 + b3 + 2*b4 + 2*b5 <= 6 and "
                                         "t1 <= b[S7] and "
                                         "p1 <= b[S2] and "
                                         "p2 <= b[S4] and "
                                         "p1 + p2 <= b[S3]"))
    pn.add_arc_by_ids("to plan", "plan", "b")
    pn.add_place(Place("to evolve"))
    pn.add_arc_by_ids("plan", "to evolve", "(b1, b2, b3, b4, b5, p1, p2, t1, b[S1], b[S2], b[S3], b[S4], b[S5], b[S6], b[S7], b[S8], b[S9], b[TIME])")
    pn.add_place(Place("b1"))
    pn.add_marks_by_id("b1", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("b1", "plan", "b1")
    pn.add_arc_by_ids("plan", "b1", "b1")
    pn.add_place(Place("b2"))
    pn.add_marks_by_id("b2", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("b2", "plan", "b2")
    pn.add_arc_by_ids("plan", "b2", "b2")
    pn.add_place(Place("b3"))
    pn.add_marks_by_id("b3", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("b3", "plan", "b3")
    pn.add_arc_by_ids("plan", "b3", "b3")
    pn.add_place(Place("b4"))
    pn.add_marks_by_id("b4", [Token(0)])
    pn.add_arc_by_ids("b4", "plan", "b4")
    pn.add_arc_by_ids("plan", "b4", "b4")
    pn.add_place(Place("b5"))
    pn.add_marks_by_id("b5", [Token(0)])
    pn.add_arc_by_ids("b5", "plan", "b5")
    pn.add_arc_by_ids("plan", "b5", "b5")
    pn.add_place(Place("p1"))
    pn.add_marks_by_id("p1", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("p1", "plan", "p1")
    pn.add_arc_by_ids("plan", "p1", "p1")
    pn.add_place(Place("p2"))
    pn.add_marks_by_id("p2", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("p2", "plan", "p2")
    pn.add_arc_by_ids("plan", "p2", "p2")
    pn.add_place(Place("t1"))
    pn.add_marks_by_id("t1", [Token(0), Token(1), Token(2), Token(3)])
    pn.add_arc_by_ids("t1", "plan", "t1")
    pn.add_arc_by_ids("plan", "t1", "t1")

    # EVOLVE
    pn.add_transition(Transition("evolve"))
    pn.add_arc_by_ids("to evolve", "evolve", "b")
    pn.add_place(Place("to demand"))
    pn.add_arc_by_ids("evolve", "to demand", "(0, 0, 0, 0, 0, 0, 0, 0, b[B2], b[B1], b[S1], b[B3], b[P1], b[P2], b[B4], b[B5], b[T1], b[TIME])")
    pn.add_place(Place("demand"))
    pn.add_arc_by_ids("evolve", "demand", "(random.choice(['S7', 'S8', 'S9']), random.choice([1, 2, 3, 4, 5, 6]))")

    # DEMAND
    pn.add_transition(Transition("process demand"))
    pn.add_arc_by_ids("to demand", "process demand", "b")
    pn.add_arc_by_ids("demand", "process demand", "d")
    pn.add_arc_by_ids("process demand", "to plan", "(0, 0, 0, 0, 0, 0, 0, 0, b[S1], b[S2], b[S3], b[S4], b[S5], b[S6], max(b[S7]-d[1], 0) if d[0]=='S7' else b[S7], max(b[S8]-d[1], 0) if d[0]=='S8' else b[S8], max(b[S9]-d[1], 0) if d[0]=='S9' else b[S9], b[TIME])")

    print(pn)
    print()

    test_run = pn.simulation_run(31)
    for test_binding in test_run:
        print(test_binding)
