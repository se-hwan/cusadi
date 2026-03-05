from typing import Callable
from abc import ABC

'''
Base interface for robot models
'''
class RobotModel():
    name = ''
    NQ = -1     # number of generalized coordinates
    NV = -1     # number of generalized velocities
    NJ = -1     # number of joints
    _functions = {}

    def __init__(self):
        pass
    
    def add_function(self, name: str, fn: Callable):
        """Attach a callable as a named function (e.g. fwd_kin, dynamics)."""
        if not callable(fn):
            raise TypeError(f"Function '{name}' must be callable.")
        self._functions[name] = fn

    def __getattr__(self, name):
        """
        Allow direct access via model.fwd_kin(...) if a function was registered.
        """
        if name in self._functions:
            return self._functions[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def has_function(self, name: str) -> bool:
        return name in self._functions