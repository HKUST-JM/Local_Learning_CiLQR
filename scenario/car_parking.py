import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelBase
from utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.transforms as tr

class CarParking(DynamicModelBase):
    """In this example, the vehicle with 4 states and 2 actions, park at (1, -1) heading to the top. 
    We hope that the vechile can park at (0, 0) and head to the right.
    The state are listed as follows: ``x0(state): position_x``, ``x1(state): position_y``, ``x2(state): heading anglue``, 
    ``x3(state): velocity``, ``x4(action): steering angle``, ``x5(action): acceleration``.
    If is_with_constraints = True, then the steering angle is limited to [-0.6, 0.6], acceleration is limited to [-3, 3].

    :param is_with_constraints: Whether the constraints are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Prediction horizon, defaults to 500
    :type T: int, optional
    """

    def __init__(self, is_with_constraints=True, T=150):
        ##### Dynamic Function ########
        n, m = 4, 2  # number of state = 4, number of action = 1, prediction horizon = 150
        h_constant = 0.1  # sampling time
        x_u_var = sp.symbols('x_u:6')
        d_constant = 3
        h_d_constanT = h_constant/d_constant
        b_function = d_constant \
            + h_constant*x_u_var[3]*sp.cos(x_u_var[4]) \
            - sp.sqrt(d_constant**2 - (h_constant**2) *
                      (x_u_var[3]**2)*(sp.sin(x_u_var[4])**2))
        dynamic_function = sp.Array([
            x_u_var[0] + b_function*sp.cos(x_u_var[2]),
            x_u_var[1] + b_function*sp.sin(x_u_var[2]),
            x_u_var[2] + sp.asin(h_d_constanT*x_u_var[3]*sp.sin(x_u_var[4])),
            x_u_var[3]+h_constant*x_u_var[5]])
        init_state = np.asarray([4, -1, np.pi/2, 0],
                                dtype=np.float64).reshape(-1, 1)
        init_action = np.zeros((T, 2, 1))
        if is_with_constraints:
            box_constr = np.asarray([[-4, np.inf], [-np.inf, np.inf],
                                 [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]])
            other_constr =  [-((x_u_var[0] + 1)**2/(3**2) + (x_u_var[1] - 4)**2/(2**2) - 1), 
                             -((x_u_var[0] + 1)**2/(3**2) + (x_u_var[1] + 4)**2/(2**2) - 1)]
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                                 [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
            other_constr = []
        ##### Objective Function ########
        switch_var = sp.symbols("s:2")
        add_param_obj = np.zeros((T, 2), dtype=np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((1., 0.))
            else:
                add_param_obj[tau] = np.asarray((0., 1.))

        def Huber_fun(x, p):
            return sp.sqrt((x**2)+(p**2)) - p

        runing_obj = Huber_fun(
            x_u_var[0]+5, 0.01) + Huber_fun(x_u_var[1], 0.01) + Huber_fun(x_u_var[2], 0.01)
        terminal_obj = 1000*Huber_fun(x_u_var[0]+5, 0.01) + 1000*Huber_fun(
            x_u_var[1], 0.01) + 1000*Huber_fun(x_u_var[2], 0.01) + 100*Huber_fun(x_u_var[3], 0.01)
        action_obj = x_u_var[4]**2 + x_u_var[5]**2
        obj_fun = switch_var[0] * runing_obj + \
            switch_var[1]*terminal_obj + action_obj
        super().__init__(dynamic_function=dynamic_function,
                         x_u_var=x_u_var,
                         box_constr=box_constr,
                         other_constr=other_constr,
                         init_state=init_state,
                         init_action=init_action,
                         obj_fun=obj_fun,
                         add_param_var=switch_var,
                         add_param=add_param_obj)

    def play(self, logger_folder=None, no_iter=-1):
        """ If ``logger_folder`` exists and the result json file is saved, 
        then the specific iteration can be chosen to play the animation.
        If ``logger_folder`` is set to be none, then the trajectroy in the last iteration will be played.

        :param logger_folder: Name of the logger folder where the result json is saved, defaults to True
        :type logger_folder: str, optional
        :param no_iter: Number of iteration to play the animation. If it is set as -1, 
            then the trajectroy in the last iteration in the given result file will be played. defaults to -1.
        :type no_iter: int, optional    
        """
        fig, ax = super().create_plot(figsize=(8, 8), xlim=(-8, 8), ylim=(-8, 8))
        plt.grid(False)
        trajectory = np.asarray(logger.read_from_json(
            logger_folder, no_iter)["trajectory"])
        
        plt.imshow(plt.imread("CuriousRL/scenario/parking_lot.png"), extent=[-8, 8, -8, 8])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'C1')
        car = plt.imshow(plt.imread("CuriousRL/scenario/vehicle.png"), extent=[0, 5, -1.5, 1.5], origin="lower")
        self._is_interrupted = False
        for i in range(self.T):
            angle = trajectory[i, 2, 0]
            t_start = ax.transData
            x = trajectory[i, 0, 0]
            y = trajectory[i, 1, 0]
            car.set_extent([x, x+5, y-1.5, y+1.5]) 
            rotate_center = t_start.transform([x, y])
            t = mpl.transforms.Affine2D().rotate_around(
                rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.001)
            if self._is_interrupted:
                return
        self._is_interrupted = True
