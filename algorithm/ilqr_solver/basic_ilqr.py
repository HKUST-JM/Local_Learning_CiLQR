from __future__ import annotations
import time as tm
from utils.Logger import logger
from .ilqr_dynamic_model import iLQRDynamicModel
from .ilqr_obj_fun import iLQRObjectiveFunction
from .ilqr_wrapper import iLQRWrapper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scenario.dynamic_model import DynamicModelBase


class BasiciLQR(iLQRWrapper):
    def __init__(self,
                 max_iter=1000,
                 is_check_stop=True,
                 stopping_criterion=1e-6,
                 max_line_search=50,
                 gamma=0.5,
                 line_search_method="vanilla",
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
                         max_iter = max_iter,
                         is_check_stop = is_check_stop)
        self._max_iter = max_iter
        self._is_check_stop = is_check_stop

    def init(self, scenario: DynamicModelBase) -> BasiciLQR:
        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.name + "\" cannot learn with LogBarrieriLQR")
        # Initialize the dynamic_model and objective function
        self._dynamic_model = iLQRDynamicModel(dynamic_function=scenario.dynamic_function,
                                              x_u_var=scenario.x_u_var,
                                              box_constr=scenario._box_constr,
                                              init_state=scenario.init_state,
                                              init_action=scenario.init_action,
                                              add_param_var=None,
                                              add_param=None)
        self._obj_fun = iLQRObjectiveFunction(obj_fun=scenario.obj_fun,
                                             x_u_var=scenario.x_u_var,
                                             add_param_var=scenario.add_param_var,
                                             add_param=scenario.add_param)

        return self

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self._trajectory = self.dynamic_model.eval_traj()  # init feasible trajectory
        C_matrix = self.obj_fun.eval_hessian_obj_fun(self._trajectory)
        c_vector = self.obj_fun.eval_grad_obj_fun(self._trajectory)
        F_matrix = self.dynamic_model.eval_grad_dynamic_model(
            self._trajectory)
        logger.info("[+ +] Initial Obj.Val.: %.5e" %
                    (self.obj_fun.eval_obj_fun(self._trajectory)))
        # Start iteration
        start_time = tm.time()
        for i in range(self._max_iter):
            if i == 1:  # skip the compiling time
                start_time = tm.time()
            iter_start_time = tm.time()
            K_matrix, k_vector = self.backward_pass(
                C_matrix, c_vector, F_matrix)
            backward_time = tm.time()
            self._trajectory, C_matrix, c_vector, F_matrix, obj, isStop = self.forward_pass(
                self._trajectory, K_matrix, k_vector)
            forward_time = tm.time()
            logger.info("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e" % (
                        i,  backward_time-iter_start_time, forward_time-backward_time, obj))
            logger.save_to_json(trajectory=self._trajectory.tolist())
            if isStop and self._is_check_stop:
                break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e" % (end_time-start_time))

    @property
    def obj_fun(self) -> iLQRObjectiveFunction:
        return self._obj_fun

    @property
    def dynamic_model(self) -> iLQRDynamicModel:
        return self._dynamic_model