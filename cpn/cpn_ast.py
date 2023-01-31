import json
from petrinet import AEPetriNet, PetriNet
from pncomponents import Place, Transition, TaggedTransition

class ASTNode:
    def visit(self, visitor):
        return visitor.accept(self)

    def __repr__(self):
        return self.__str__()


class ExpNode(ASTNode):
    def __init__(self, expression):
        self.expression = expression


class PetriNetNode(ASTNode):
    def __init__(self, places, transitions, arcs, marking):
        self.places = places
        self.transitions = transitions
        self.arcs = arcs
        self.marking = marking

    def __str__(self):
        result = ""
        result += "Places:\n"
        for p in self.places:
            result += str(p) + "\n"
        result += "\nTransitions:\n"
        for t in self.transitions:
            result += str(t) + "\n"
        result += "\nArcs:\n"
        for a in self.arcs:
            result += str(a) + "\n"
        result += "\nMarking:\n"
        for m in self.marking:
            result += str(m) + "\n"
        return result


class PlaceNode(ASTNode):
    def __init__(self, p_id, label):
        self.p_id = p_id
        self.label = label

    def __str__(self):
        return self.p_id


class TransitionNode(ASTNode):
    def __init__(self, t_id, tag=None, reward=None):
        self.t_id = t_id
        self.tag = tag
        self.reward= reward

    def __str__(self):
        return self.t_id


class ArcNode(ASTNode):
    def __init__(self, src_id, dst_id, variable=None, tw_low=None, tw_high=None, increment=None, delay_type=None, params=None):
        self.src_id = src_id
        self.dst_id = dst_id
        self.variable = variable
        self.tw = (tw_low, tw_high)
        self.increment = increment
        self.delay_type = delay_type
        self.params = params

    def __str__(self):
        result = "("
        result += self.src_id + "," + self.dst_id
        if self.variable is not None:
            result += "," + self.variable
        if self.tw is not None:
            result += "," + self.tw
        if self.increment is not None:
            result += ",@t+" + self.increment
        result += ")"
        return result


class MarkingNode(ASTNode):
    def __init__(self, p_id, tokens):
        self.p_id = p_id
        self.tokens = tokens

    def __str__(self):
        result = "(" + self.p_id + ","
        for i in range(len(self.tokens)):
            result += str(self.tokens[i])
            if i < len(self.tokens) - 1:
                result += "++"
        result += ")"
        return result

#TODO: composite colors in initial markings are not considered
class TokenNode(ASTNode):
    def __init__(self, count, key_value):
        self.count = count
        key, values = key_value.split(':')
        values = values.split(';')
        self.color = {key : values}

    def __str__(self):
        return str(self.count) + "`" + self.color + "`"


class PetriNetCreatorVisitor:
    def __init__(self, net_type="PetriNet"):
        if net_type == "PetriNet":
            self.pn = PetriNet()
        elif net_type == "AEPetriNet":
            self.pn = AEPetriNet()
        else:
            raise Exception("Invalid net type")

    def accept(self, element):
        if type(element).__name__ == "PetriNetNode":
            for place_node in element.places:
                place_node.visit(self)
            for transition_node in element.transitions:
                transition_node.visit(self)
            for arc_node in element.arcs:
                arc_node.visit(self)
            for marking_node in element.marking:
                marking_node.visit(self)
        elif type(element).__name__ == "PlaceNode":
            self.pn.add_place(Place(element.p_id))
        elif type(element).__name__ == "TransitionNode":
            if element.tag is None:
                self.pn.add_transition(Transition(element.t_id))
            else:
                self.pn.add_transition(TaggedTransition(element.t_id, element.tag, element.reward))

        elif type(element).__name__ == "ArcNode":
            self.pn.add_arc_by_ids(element.src_id, element.dst_id, element.variable, element.tw, element.increment, element.delay_type, element.params)
        elif type(element).__name__ == "MarkingNode":
            tokens = dict()
            for token_node in element.tokens:
                (value, count) = token_node.visit(self)
                tokens[json.dumps(value)] = count #json module is used to obtain a string representation of the list and easily convert it back to list when needed
            self.pn.add_mark_by_id(element.p_id, tokens)
        elif type(element).__name__ == "TokenNode":
            return element.color, element.count

    def create(self, petrinet_node):
        petrinet_node.visit(self)
        return self.pn
