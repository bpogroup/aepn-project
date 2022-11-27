from cpn_ast import PetriNetCreatorVisitor
from pn_parser import PNParser
from petrinet import PetriNet, Place, Transition


# Test run on classical Petri net
print()
print("TEST CLASSICAL PETRI NET")
print()

pn = PetriNet()

pn.add_place(Place("arrival"))
pn.add_place(Place("waiting"))
pn.add_place(Place("resources"))
pn.add_place(Place("busy"))

pn.add_transition(Transition("arrive"))
pn.add_transition(Transition("start"))
pn.add_transition(Transition("complete"))

pn.add_arc_by_ids("arrival", "arrive")
pn.add_arc_by_ids("arrive", "arrival")
pn.add_arc_by_ids("arrive", "waiting")
pn.add_arc_by_ids("waiting", "start")
pn.add_arc_by_ids("resources", "start")
pn.add_arc_by_ids("start", "busy")
pn.add_arc_by_ids("busy", "complete")
pn.add_arc_by_ids("complete", "resources")

pn.add_mark_by_id("resources", {None: 2})
pn.add_mark_by_id("arrival", {None: 1})

run = pn.simulation_run(10)
for binding in run:
    print(binding)

print(pn)

# Test run on colored Petri net:
# resources can only pick up cases with same color token
# there are two colors: a and b
print()
print("TEST PARSER AND COLORED PETRI NET")
print()

f = open("mynet.txt", "r")
mynet_txt = f.read()

pn_ast = PNParser().parse(mynet_txt)
pn = PetriNetCreatorVisitor().create(pn_ast)
print(pn)
print()

pn = PetriNet()

pn.add_place(Place("arrival"))
pn.add_place(Place("waiting"))
pn.add_place(Place("resources"))
pn.add_place(Place("busy"))

pn.add_transition(Transition("arrive"))
pn.add_transition(Transition("start"))
pn.add_transition(Transition("complete"))

pn.add_arc_by_ids("arrival", "arrive", "x")
pn.add_arc_by_ids("arrive", "arrival", "x")
pn.add_arc_by_ids("arrive", "waiting", "x")
pn.add_arc_by_ids("waiting", "start", "x")
pn.add_arc_by_ids("resources", "start", "x")
pn.add_arc_by_ids("start", "busy", "x")
pn.add_arc_by_ids("busy", "complete", "x")
pn.add_arc_by_ids("complete", "resources", "x")

pn.add_mark_by_id("resources", {"a": 1, "b": 1})
pn.add_mark_by_id("arrival", {"a": 1, "b": 1})

print(pn)
print()

run = pn.simulation_run(10)
for binding in run:
    print(binding)
