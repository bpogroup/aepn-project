# Assignment Graphs as Universal Feature Representation Approach for Dynamic Task Assignment Problems
This package contains the Python implementation of graph assignments as observations for Action-Evolution Petri Nets, an approach to feature representation in task assignment problems that allows modeling finite and infinite state/actions spaces and learn policies to solve them through DRL.

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
python __main__.py
```
If the installation was successful, you will see a stream of outputs indicating that PPO is being trained on a simple task assignment problem.

## Usage
The package is designed to be used as a library, and the main entry point is the `__main__.py` file. By default, the script will train PPO on a small example problem.
To change problem, modify the `__main__.py` file to use the desired problem and run the script. The same is valid to use different solving strategies, or to use a trained policy at inference time.