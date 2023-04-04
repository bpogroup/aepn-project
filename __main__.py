import os
import sys
import argparse
import copy
import numpy as np
from cpn.cpn_ast import PetriNetCreatorVisitor
from cpn.pn_parser import PNParser
from cpn.petrinet import PetriNet, AEPetriNet, Place, Transition



def main(args):
    #parse net file
    f = open(os.path.join('networks', args.filename), 'r')
    mynet_txt = f.read()
    f.close()
    pn_ast = PNParser().parse(mynet_txt)

    #parse additional_functions
    f = open(os.path.join('cpn', 'color_functions.py'), 'r') #read functions file as txt
    my_functions = f.read()
    f.close()

    #create pn
    pn = PetriNetCreatorVisitor(net_type="AEPetriNet", additional_functions=my_functions).create(pn_ast)
    print(pn)

    #fixed values for test run (to be changed to be input)
    episode_length = 100
    num_episodes = 1000

    train_ae_pn = args.train if args.train else True
    load_ppo = args.load if args.load else False
    test_ae_pn = args.test if args.test else True


    if train_ae_pn:
        # Test run on A-E Petri net:
        run = pn.training_run(100000, 100, load_ppo)
    
    if test_ae_pn:
        r_vec_ppo = []
        r_vec_random = []
        for i in range(num_episodes):
            pn = copy.copy(pn)
            rand_pn = copy.deepcopy(pn)

            total_reward_ppo = pn.testing_run(episode_length, additional_functions = my_functions)
            r_vec_ppo.append(total_reward_ppo)
            run, total_reward_random = rand_pn.simulation_run(episode_length)
            r_vec_random.append(total_reward_random)
        import numpy as np
        print(f"Average reward PPO: {np.mean(r_vec_ppo)} with standard deviation {np.std(r_vec_ppo)}\nAverage reward random: {np.mean(r_vec_random)} with standard deviation {np.std(r_vec_random)}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--filename", help="the name of the file where the desired A-E PN is defined (with extension)")
    argParser.add_argument("--train", help="boolean indicating wether to train a new network")
    argParser.add_argument("--load", help="boolean indicating wether to load a trained network (for training, ignored if train is False)")
    argParser.add_argument("--test", help="boolean indicating wether to test a trained network")
    args = argParser.parse_args()
    main(args)