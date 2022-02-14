#%%
import numpy as np 
import sympy as sp 
from numba import njit 
from utils.Logger import logger 
import matplotlib.pyplot as plt 

class DynamicModelBase():
    """ In this example, the cartpole system is static at 0, 0, heading to the postive direction of the y axis\\
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right\\
        x0: angle, x1: angular velocity, x2: position, x3: velocity, x4: force
    """
    def __init__(self, 
                dynamic_function, 
                x_u_var, 
                box_constr, 
                init_state, 
                init_action, 
                obj_fun, 
                add_param_var = None, 
                add_param = None,
                other_constr = []): 
        self._dynamic_function = dynamic_function
        self._x_u_var = x_u_var
        self._init_state = init_state
        self._init_action = init_action
        self._box_constr = box_constr
        self._other_constr = other_constr
        self._n = int(init_state.shape[0])
        self._m = int(len(x_u_var) - self._n)
        self._T = int(init_action.shape[0])
        self._obj_fun = obj_fun
        self._add_param_var = add_param_var
        self._add_param = add_param
        self._fig = None
    
    def create_plot(self, figsize =(5, 5), xlim = (-6,6), ylim = (-6,6), zlim = (-6,6), is_3d = False, is_equal = True):
        if self._fig == None:
            self._fig = plt.figure(figsize = figsize)
            if not is_3d:
                self._ax = self._fig.add_subplot(111)
            else:
                self._ax = self._fig.add_subplot(111, projection='3d')
            if is_equal:
                self._ax.axis('equal')
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            if is_3d:
                self._ax.set_zlim(*zlim)    
            self._ax.grid(True)
        return self._fig, self._ax

    def with_model(self):
        return True

    def is_action_discrete(self):
        return False

    def is_output_image(self):
        return False

    @property
    def dynamic_function(self):
        return self._dynamic_function

    @property
    def x_u_var(self):
        return self._x_u_var
        
    @property
    def init_state(self):
        return self._init_state

    @property
    def init_action(self):
        return self._init_action

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def T(self):
        return self._T

    @property
    def obj_fun(self):
        return self._obj_fun

    @property
    def add_param_var(self):
        return self._add_param_var

    @property
    def add_param(self):
        return self._add_param 
    
    @property
    def box_constr(self):
        return self._box_constr

    @property
    def other_constr(self):
        return self._other_constr

    @property
    def name(self):
        return self.__class__.__name__

    def play(self, logger_folder = None, no_iter = -1):
        raise NotImplementedError