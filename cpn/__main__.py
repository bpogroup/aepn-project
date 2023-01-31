from cpn_ast import PetriNetCreatorVisitor
from pn_parser import PNParser
from petrinet import PetriNet, AEPetriNet, Place, Transition
import numpy as np


test_classical_pn = False

test_colored_pn = False

train_ae_pn = True
load_ppo = False

test_ae_pn = True


if test_classical_pn:
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


if test_colored_pn:
    # Test run on colored Petri net:
    # resources can only pick up cases with same color token
    # there are two colors: a and b
    print()
    print("TEST COLORED PETRI NET")
    print()

    f = open("cpn/mynet.txt", "r")
    aenet_txt = f.read()

    pn_ast = PNParser().parse(aenet_txt)
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

if train_ae_pn:
    # Test run on A-E Petri net:
    # resources can only pick up cases with same color token, transitions are tagged and arcs are timed
    # there are three colors: a, b and a;b (the latter meaning it can be combined with any a or b)
    print()
    print("TEST A-E PETRI NET PARSER")
    print()

    f = open("cpn/aenet.txt", "r")
    mynet_txt = f.read()
    pn_ast = PNParser().parse(mynet_txt)
    pn = PetriNetCreatorVisitor(net_type="AEPetriNet").create(pn_ast)
    run = pn.training_run(10, 1000, load_ppo)
    
if test_ae_pn:
    rew_ppo_vec = []
    rew_random_vec = []
    f = open("cpn/aenet.txt", "r")
    mynet_txt = f.read()
    for i in range(1):
        pn_ast = PNParser().parse(mynet_txt)
        pn = PetriNetCreatorVisitor(net_type="AEPetriNet").create(pn_ast)

        rew_ppo = pn.testing_run(100)
        rew_ppo_vec.append(rew_ppo)
        pn = PetriNetCreatorVisitor(net_type="AEPetriNet").create(pn_ast)
        run, rew_random = pn.simulation_run(100)
        rew_random_vec.append(rew_random)

    print(f"Average reward ppo: {np.mean(rew_ppo_vec)}")
    print(f"Average reward random: {np.mean(rew_random_vec)}")
    