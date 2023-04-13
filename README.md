# Action-Evolution Petri Nets
This package contains the Python implementation of Action-Evolution Petri Nets, a framework for modeling and solving assignment problems. The package allows to define new assignment problems using the Action-Evolution and to learn a policy for solving the problem through PPO, a Deep Reinforcement Learning algorithm.

## Installation
This guide assumes the user has a working Python 3 installation available, including the pip package manager.
Download the repository, then open a bash in the main folder and create/activate a Python virtual environment by running the command

```
python -m venv env
.\env\Scripts\activate
```
Proceed by installing the required packages from `requirements.txt`
```
python -m pip install -r requirements.txt
```
To test the installation, run the command
```
python __main__.py --filename aenet.txt
```
If the installation was successful, you will see a stream of outputs indicating that PPO is being trained on a simple task assignment problem.

## Training a solver
The syntax for trainining a new solver is as follows
```
python __main__.py --filename PROBLEM_FILE.txt --train True
```
It is also possible to retrieve a previously trained algorithm to resume the training
```
python __main__.py --filename PROBLEM_FILE.txt --train True --load True
```

## Testing a solver
After a solver has been trained, run the following command to check its average reward over a set of episodes (compared against a random policy)
```
python __main__.py --filename PROBLEM_FILE.txt --train False --test True
```
It is possible to train a new algorithm and test it right after the end of training
```
python __main__.py --filename PROBLEM_FILE.txt --train True --test True
```

## Environments
A set of assignment problems with different characteristics were modeled in A-E PN notation as plain text files and made available in the `networks` folder. The problems that were modeled until now are:
+ **Task Assignment Problem with Hard Compatibilities (_task\_assignment\_hard\_comp.txt_)**
+ **Task Assignment Problem with Soft Compatibilities (_task\_assignment\_soft\_comp.txt_)**
+ **Dynamic Bin Packing Problem (_bin\_packing.txt_)**
+ **Order Picking Problem (_order\_picking.txt_)**
+ **Task Assignment Problem with Leaving Resources (_task\_assignment\_leaving\_resources.txt_)**
+ **Collaborative Task Assignment Problem (_task\_assignment\_collaborative.txt_)**

## Remarks
The color-specific functions used as guards for firing transitions and as reward functions are currently collected in `additional_functions.py`. In a future release, they will be included in the problem definitions.
