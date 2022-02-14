from __future__ import annotations
import numpy as np
import sympy as sp
import time as tm
from utils.Logger import logger
from .basic_ilqr import iLQRWrapper
from .ilqr_obj_fun import iLQRObjectiveFunction
from .ilqr_dynamic_model import iLQRDynamicModel
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scenario.dynamic_model import DynamicModelBase

class LogBarrieriLQR(iLQRWrapper):
    def __init__(self,
                 max_iter=1000,
                 is_check_stop=True,
                 stopping_criterion=1e-6,
                 max_line_search=50,
                 gamma=0.5,
                 t=[0.5, 1., 2., 5., 10., 20., 50., 100.],
                 line_search_method="feasibility",
                 stopping_method="relative"):
        """
            Parameter
            -----------
            :param max_iter: Maximum number of the iLQR iterations.
            :type max_iter: int
            :param is_check_stop: Decide whether the stopping criterion is checked.
                If is_check_stop = False, then the maximum number of the iLQR iterations will be reached.
            :type is_check_stop: bool
            is_check_stop : boolean
                Whether the stopping criterion is checked
            stopping_criterion : double
                The stopping criterion
            max_line_search : int
                Maximum number of line search
            gamma : double 
                Gamma is the parameter for the line search: alpha=gamma*alpha
            line_search_method : str
            stopping_method : str
        """
        super().__init__(stopping_criterion=stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method,
                         max_iter=max_iter,
                         is_check_stop=is_check_stop)
        self._t = t
        self._max_iter = max_iter
        self._is_check_stop = is_check_stop

    def init(self, scenario: DynamicModelBase) -> LogBarrieriLQR:
        """ Initialize the iLQR solver class

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR

            Return 
            LogBarrieriLQR
        """
        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.name + "\" cannot learn with LogBarrieriLQR")
        # Parameters for the model
        self._dynamic_model = iLQRDynamicModel(dynamic_function=scenario.dynamic_function,
                                               x_u_var=scenario.x_u_var,
                                               box_constr=scenario.box_constr,
                                               init_state=scenario.init_state,
                                               init_action=scenario.init_action,
                                               add_param_var=None,
                                               add_param=None)
        self._real_obj_fun = iLQRObjectiveFunction(obj_fun=scenario.obj_fun,
                                                   x_u_var=scenario.x_u_var,
                                                   add_param_var=scenario.add_param_var,
                                                   add_param=scenario.add_param)
        x_u_var = scenario.x_u_var
        t_var = sp.symbols('t')  # introduce the parameter for log barrier
        add_param_var = scenario.add_param_var
        if add_param_var is None:
            add_param_var = (t_var,)
        else:
            add_param_var = (*add_param_var, t_var)
        # construct the barrier objective function
        barrier_obj_fun = scenario.obj_fun
        # add the inequality constraints to the objective function
        for i, c in enumerate(scenario.box_constr):
            if not np.isinf(c[0]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(c[0] - x_u_var[i]))
            if not np.isinf(c[1]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(x_u_var[i] - c[1]))
        for other_constr in scenario._other_constr:
            barrier_obj_fun += (-1/t_var)*sp.log(-other_constr)
        self._obj_fun = iLQRObjectiveFunction(obj_fun=barrier_obj_fun,
                                              x_u_var=x_u_var,
                                              add_param_var=add_param_var)
        self.set_obj_add_param(scenario.add_param)
        return self

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self._trajectory = self.dynamic_model.eval_traj()  # init feasible trajectory
        logger.info(f"[+ +] Init State: \n{self._dynamic_model._init_state}")
        C_matrix = self.obj_fun.eval_hessian_obj_fun(self._trajectory)
        c_vector = self.obj_fun.eval_grad_obj_fun(self._trajectory)
        F_matrix = self.dynamic_model.eval_grad_dynamic_model(
            self._trajectory)
        # Start iteration
        logger.info("[+ +] Initial Obj.Val.: %.5e" %
                    (self._real_obj_fun.eval_obj_fun(self._trajectory)))
        total_iter_no = -1
        for idx_j, j in enumerate(self._t):
            self.set_obj_add_param(t_index=idx_j)
            self.set_obj_fun_value(self._obj_fun.eval_obj_fun(self._trajectory))
            for i in range(self._max_iter):
                total_iter_no += 1
                if idx_j == 0 and i == 1:  # skip the compiling time
                    start_time = tm.time()
                iter_start_time = tm.time()
                K_matrix, k_vector = self.backward_pass(
                    C_matrix, c_vector, F_matrix)
                backward_time = tm.time()
                self._trajectory, C_matrix, c_vector, F_matrix, _, isStop = self.forward_pass(
                    self._trajectory, K_matrix, k_vector)
                forward_time = tm.time()
                # do not care the value of log barrier
                obj = self._real_obj_fun.eval_obj_fun(self._trajectory)
                logger.info("[+ +] Total Iter.No.%3d   Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e" % (
                    total_iter_no,     i,  backward_time-iter_start_time, forward_time-backward_time, obj))
                logger.save_to_json(trajectory=self._trajectory.tolist())
                if isStop and self._is_check_stop:
                    
                    logger.info(
                        "[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f" % (j) + " in this iteration!")
                    break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e" % (end_time-start_time))

    @property
    def obj_fun(self) -> iLQRObjectiveFunction:
        return self._obj_fun

    @property
    def dynamic_model(self) -> iLQRDynamicModel:
        return self._dynamic_model

    def set_obj_add_param(self, new_add_param = None, t_index = None):
        """Set the values of the additional parameters in the objective function

        :param new_add_param: The new values to the additioanl variables
        :type new_add_param: array(T, p)
        """
        t = self._t[0 if t_index is None else t_index]
        if self._real_obj_fun._add_param is None:
            add_param =  t * \
                np.ones((self.dynamic_model._T, 1), dtype=np.float64)
        else:
            if new_add_param is not None:
                self._real_obj_fun._add_param = new_add_param
            add_param = np.hstack([self._real_obj_fun._add_param, t*np.ones((self.dynamic_model._T, 1))])
        self._obj_fun._add_param = add_param
        
