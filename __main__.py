import copy
import platform
import time

import numpy as np
import cpn.new_petrinet as new_petrinet

from cpn.additional_functions.new_color_functions import generate_vector_color_set
from cpn.pncomponents import TaggedTransition, Place

if __name__ == "__main__":

    observation_type = 'graph' #'vector' for old version, 'graph' for new version

    test_simple_aepn = False #test a simple aepn for testing purposes
    test_task_assignment = False #test a-e cpn for task assignment problem
    test_simple_consulting_firm = True
    test_consulting_firm = False
    test_hard_consulting_firm = False

    test_simulation = False
    test_training = True
    test_inference = False
    test_heuristic = False

    #maximum value of self.clock
    max_length = 9 #this is actually 10 steps since we start from 0

    #for testing
    replications = 1000

    if platform.system() == 'Windows':
        # color functions
        f = open('cpn/additional_functions/new_color_functions.py', 'r')
        temp = f.read()
        f.close()

        # time functions
        f_t = open('cpn/additional_functions/time_functions.py', 'r')
        temp_t = f_t.read()
        f_t.close()
    elif platform.system() == 'Linux':
        # color functions
        f = open('/new_color_functions.py', 'r')
        temp = f.read()
        f.close()

        # time functions
        f_t = open('/time_functions.py', 'r')
        temp_t = f_t.read()
        f_t.close()
    else:
        raise Exception("OS not supported")


    my_functions = temp + '\n' + temp_t
    pn = new_petrinet.AEPetriNet(my_functions, observation_type=observation_type)

    if observation_type == 'vector':
        if test_simple_consulting_firm:
            pn.add_place(Place("arrival", colors_set={'{"type":0,"budget":100}','{"type":1,"budget":200}'})) #CAREFUL: due to a piece of code in get_actions, if case is defined before resource, as a place, also the arcs need to be in the same order (see HERE)
            pn.add_place(Place("waiting", colors_set={'{"type":0,"budget":100}','{"type":1,"budget":200}'}))
            pn.add_place(Place("resources", colors_set={'{"id":0}'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward='get_budget_str'))
            # pn.add_transition(TaggedTransition("complete", tag='e', reward='get_budget'))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "x", time_expression=0)

            pn.add_arc_by_ids("waiting", "start", "case") #HERE if this and the next line are switched, it breaks. TODO: modify get_actions
            pn.add_arc_by_ids("resources", "start", "resource")

            pn.add_arc_by_ids("start", "resources", "resource", time_expression=1)

            pn.add_mark_by_id("arrival", '{"type":0,"budget":100}', 0) #CAREFUL: no spaces!!
            pn.add_mark_by_id("arrival", '{"type":1,"budget":200}', 0)
            pn.add_mark_by_id("resources", '{"id":0}', 0)  # HERE: same as before!

        elif test_consulting_firm:
            pn.add_place(Place("arrival", colors_set={'{"type":0}','{"type":1}'}))
            pn.add_place(Place("waiting", colors_set=generate_vector_color_set()))
            pn.add_place(Place("resources", colors_set={'{"id":0}'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward='get_budget_str'))
            # pn.add_transition(TaggedTransition("complete", tag='e', reward='get_budget'))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "randomize_budget_uniform_str", time_expression=0)

            pn.add_arc_by_ids("waiting", "start",
                              "case")  # HERE if this and the next line are switched, it breaks. TODO: modify get_actions
            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("start", "resources", "resource", time_expression=1)

            pn.add_mark_by_id("arrival", '{"type":0}', 0)  # CAREFUL: no spaces!!
            pn.add_mark_by_id("arrival", '{"type":1}', 0)
            pn.add_mark_by_id("resources", '{"id":0}', 0)  # HERE: same as before!

        if test_training:
            start_time = time.time()
            pn.training_run(100000, max_length, False)
            print("TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))
        elif test_inference:

            r_vec_ppo = []
            r_vec_random = []
            for i in range(replications):
                ppo_pn = copy.deepcopy(pn)
                rand_pn = copy.deepcopy(pn)

                total_reward_ppo = ppo_pn.testing_run(max_length)
                r_vec_ppo.append(total_reward_ppo)
                run, total_reward_random = rand_pn.simulation_run(max_length)
                r_vec_random.append(total_reward_random)
            import numpy as np

            print(f"Average reward PPO: {np.mean(r_vec_ppo)} with standard deviation {np.std(r_vec_ppo)}"
                  f"\nAverage reward random: {np.mean(r_vec_random)} with standard deviation {np.std(r_vec_random)}")




    else:
        if test_task_assignment:

            pn.add_place(Place("arrival", tokens_attributes={"type": "int"})) #attribute TIME is always added by default
            pn.add_place(Place("waiting", tokens_attributes={"type": "int"}))
            pn.add_place(Place("resources", tokens_attributes={"type": "int", "compatibility_0": "int", "compatibility_1": "int", "is_set": "int"}))
            pn.add_place(Place("busy", tokens_attributes={'resource_type': 'int', 'task_type': 'int', 'compatibility_0': 'int', 'compatibility_1': 'int'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("set_compatibility", guard="is_not_set", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward=0))
            pn.add_transition(TaggedTransition("complete", tag='e', reward=1))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "x", time_expression=0)

            pn.add_arc_by_ids("resources", "set_compatibility", "resource")
            pn.add_arc_by_ids("set_compatibility", "resources", "set_compatibility", time_expression=0)

            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("waiting", "start", "case")
            pn.add_arc_by_ids("start", "busy", "new_combine", time_expression="get_compatibility")

            pn.add_arc_by_ids("busy", "complete", "case_resource")
            pn.add_arc_by_ids("complete", "resources", "new_split", time_expression=0)

            pn.add_mark_by_id("resources", {'type': 0, 'compatibility_0': 1, 'compatibility_1': 1, 'is_set': 0}, 0)
            pn.add_mark_by_id("resources", {'type': 1, 'compatibility_0': 1, 'compatibility_1': 1, 'is_set': 0}, 0)
            pn.add_mark_by_id("arrival", {'type': 0}, 0)
            pn.add_mark_by_id("arrival", {'type': 1}, 0)

        #print(pn)
        elif test_simple_aepn:
            #orders could be lost if they are not processed within 1 time unit
            pn.add_place(Place("arrival", tokens_attributes={"type": "int"}))
            pn.add_place(Place("waiting", tokens_attributes={"type": "int"}))
            pn.add_place(Place("resources", tokens_attributes={"type": "int"}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward="get_reward"))
            #pn.add_transition(TaggedTransition("lose_order", tag='e', reward=0))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "x", time_expression=0)
            #pn.add_arc_by_ids("waiting", "lose_order", "x", guard_function="is_late(x)", time_expression=0)
            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("waiting", "start", "case")
            pn.add_arc_by_ids("start", "resources", "resource", time_expression=1)

            pn.add_mark_by_id("resources", {'type': 0}, 0)
            pn.add_mark_by_id("resources", {'type': 10}, 0)
            pn.add_mark_by_id("arrival", {'type': 0}, 0)
            pn.add_mark_by_id("arrival", {'type': 10}, 0)

        elif test_simple_consulting_firm:

            pn.add_place(Place("arrival", tokens_attributes={"type": "int", "budget": "int"})) #attribute TIME is always added by default
            pn.add_place(Place("waiting", tokens_attributes={"type": "int", "budget": "int"})) #TODO: make "patience" unobservable
            pn.add_place(Place("resources", tokens_attributes={"id": "int"}))
            #pn.add_place(Place("busy", tokens_attributes={'type': 'int', 'budget': 'int'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward='get_budget'))
            #pn.add_transition(TaggedTransition("complete", tag='e', reward='get_budget'))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "x", time_expression=0)

            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("waiting", "start", "case")
            pn.add_arc_by_ids("start", "resources", "resource", time_expression=1)
            #pn.add_arc_by_ids("start", "busy", "case", time_expression=1)

            #pn.add_arc_by_ids("busy", "complete", "case_resource")
            #pn.add_arc_by_ids("complete", "resources", "empty_token", time_expression=0)

            pn.add_mark_by_id("resources", {"id": 0}, 0) #resources do not have attributes
            #pn.add_mark_by_id("resources", {'average_completion_time': 3}, 0)
            pn.add_mark_by_id("arrival", {"type": 0, 'budget': 100}, 0)
            pn.add_mark_by_id("arrival", {"type": 1, 'budget': 200}, 0)

        elif test_consulting_firm:

            pn.add_place(Place("arrival", tokens_attributes={"type": "int"})) #attribute TIME is always added by default
            pn.add_place(Place("waiting", tokens_attributes={"type": "int", "budget": "int"})) #TODO: make "patience" unobservable
            pn.add_place(Place("resources", tokens_attributes={"id": "int"}))
            #pn.add_place(Place("busy", tokens_attributes={'type': 'int', 'budget': 'int'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward='get_budget'))
            #pn.add_transition(TaggedTransition("complete", tag='e', reward='get_budget'))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression=1)
            pn.add_arc_by_ids("arrive", "waiting", "randomize_budget_uniform", time_expression=0)

            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("waiting", "start", "case")
            pn.add_arc_by_ids("start", "resources", "resource", time_expression=1)
            #pn.add_arc_by_ids("start", "busy", "case", time_expression=1)

            #pn.add_arc_by_ids("busy", "complete", "case_resource")
            #pn.add_arc_by_ids("complete", "resources", "empty_token", time_expression=0)

            pn.add_mark_by_id("resources", {"id": 0}, 0) #resources do not have attributes
            #pn.add_mark_by_id("resources", {'average_completion_time': 3}, 0)
            pn.add_mark_by_id("arrival", {'type': 0}, 0)
            pn.add_mark_by_id("arrival", {'type': 1}, 0)

        elif test_hard_consulting_firm:

            pn.add_place(Place("arrival", tokens_attributes={"type": "int", "average_interarrival_time": "int"})) #attribute TIME is always added by default
            pn.add_place(Place("waiting", tokens_attributes={"type": "int",  "budget": "int", "patience": "int", "arrival_time": "int"})) #TODO: make "patience" unobservable
            pn.add_place(Place("resources", tokens_attributes={"average_completion_time": "int"}))
            #pn.add_place(Place("busy", tokens_attributes={'resource_average_completion_time': 'int', 'task_type': 'int', 'budget': 'int'}))

            pn.add_transition(TaggedTransition("arrive", tag='e', reward=0))
            pn.add_transition(TaggedTransition("start", tag='a', reward='get_budget'))
            #pn.add_transition(TaggedTransition("complete", tag='e', reward=0))
            pn.add_transition(TaggedTransition("lose_order", tag='e', reward=0, guard="is_late"))

            pn.add_arc_by_ids("arrival", "arrive", "x")
            pn.add_arc_by_ids("arrive", "arrival", "x", time_expression='exponential_mean')
            pn.add_arc_by_ids("arrive", "waiting", "randomize_budget_and_patience", time_expression=0)

            pn.add_arc_by_ids("waiting", "lose_order", "case")

            pn.add_arc_by_ids("resources", "start", "resource")
            pn.add_arc_by_ids("waiting", "start", "case")
            #pn.add_arc_by_ids("start", "busy", "combine_budget", time_expression="randomize_completion_time") #for now, the randomization is not done and the average time is considered
            pn.add_arc_by_ids("start", "resources", "resource", time_expression="randomize_completion_time")

            #pn.add_arc_by_ids("busy", "complete", "case_resource")
            #pn.add_arc_by_ids("complete", "resources", "get_resource", time_expression=0)

            pn.add_mark_by_id("resources", {'average_completion_time': 2}, 0)
            pn.add_mark_by_id("resources", {'average_completion_time': 1}, 0)
            pn.add_mark_by_id("arrival", {'type': 0, 'average_interarrival_time': 1}, 0)
            pn.add_mark_by_id("arrival", {'type': 1, 'average_interarrival_time': 1}, 0)




        if test_simulation:
            test_run = pn.simulation_run(1000)
            for test_binding in test_run:
                print(test_binding)

        elif test_training:
            start_time = time.time()
            pn.new_training_run(max_length, test_inference)
            print("TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))

        elif test_inference:



            r_vec_ppo = []
            r_vec_heuristic = []
            r_vec_random = []
            for i in range(replications):
                ppo_pn = copy.deepcopy(pn)
                heur_pn = copy.deepcopy(pn)
                rand_pn = copy.deepcopy(pn)

                total_reward_ppo = ppo_pn.new_testing_run(max_length, additional_functions = my_functions)
                r_vec_ppo.append(total_reward_ppo)
                total_reward_heuristic = heur_pn.heuristic_testing_run(max_length)
                r_vec_heuristic.append(total_reward_heuristic)
                run, total_reward_random = rand_pn.simulation_run(max_length)
                r_vec_random.append(total_reward_random)
            import numpy as np
            print(f"Average reward PPO: {np.mean(r_vec_ppo)} with standard deviation {np.std(r_vec_ppo)}"
                  f"\nAverage reward SPT heuristic: {np.mean(r_vec_heuristic)} with standard deviation {np.std(r_vec_heuristic)}"
                  f"\nAverage reward random: {np.mean(r_vec_random)} with standard deviation {np.std(r_vec_random)}")

        elif test_heuristic:
            r_vec_heuristic = []
            r_vec_heuristic.append(pn.heuristic_testing_run(max_length))

            print(
                f"Average reward SPT heuristic: {np.mean(r_vec_heuristic)} with standard deviation {np.std(r_vec_heuristic)}")