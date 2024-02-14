#the FunctionDispatcher class is used to dispatch the function to the corresponding function in either new_color_functions.py or time_functions.py
#it is used in the following way:
#     f = FunctionDispatcher()
#     f.dispatch("check_remaining_space", object, bin)
#     """

import importlib
import inspect

class FunctionDispatcher:
#    def __init__(self):
#        self.functions = color_functions
#        self.time_functions = time_functions

    def __init__(self, module_names, clock):
        self.clock = clock
        self.commands = {}
        for module_name in module_names:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr):
                    self.commands[attr_name] = attr

    def dispatch(self, command_name, clock=None, **args):
        #import pdb; pdb.set_trace()
        if command_name in args.keys():
            return args[command_name] #if the input identity function
        elif command_name in self.commands:
            if inspect.getfullargspec(self.commands[command_name]).args[0] == 'CLOCK': #if the function has a clock argument
                return self.commands[command_name](self.clock.value, **args)
            else:
                return self.commands[command_name](**args)
        else:
            raise Exception(f"Function {command_name} not found in available functions: {self.commands.keys()}")

    def get_commands(self):
        return self.commands.keys()

if __name__ == "__main__":
    #test the FunctionDispatcher class
    f = FunctionDispatcher(['new_color_functions', 'time_functions'], 0)
    print('test')
    for func in f.get_commands():
        print(func)

    print(f.dispatch("is_not_set", {"is_set": 0}))
