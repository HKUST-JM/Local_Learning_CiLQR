from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numba import njit
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ilqr_obj_fun import iLQRObjectiveFunction
    from .ilqr_dynamic_model import iLQRDynamicModel


class iLQRWrapper():
    """This is a wrapper class for all the iLQR algorithms. 
    This class provides several basic methods for the iLQR algorithm realization. 
    This class cannot create algorithm instance. 
    To create an instance for the child class, the abstract method ``solve``, abstract property ``obj_fun``, ``dynamic_model`` are required to be realized.
    To initialize the fundamental parameters, all child classes must call ``super().__init__(...)``.

    :param stopping_criterion:  Stopping criterion for the stopping method
    :type stopping_criterion: float
    :param max_line_search:  Maximum number of line search iterations.
    :type max_line_search: int
    :param gamma: Gamma is the parameter for the line search, that is alpha=gamma*alpha if line search 
        requirements are not satisfied, where alpha is the step size of the current iteration.
    :type gamma: float
    :param line_search_method: The method for line search. The provided line search methods are listed as bellow.
        ``"vanilla"``: The current objective function value is smaller than the last
        ``"feasibility"``: The current objective function value is smaller than the last and the trajectory is feasible
        ``"none"``: No line search
    :type line_search_method: str
    :param stopping_method: The method for stopping the iteration. The provided stopping methods are listed as bellow.
        ``"vanilla"``: The difference between the last objective function value and the current objective function value is smaller than the stopping_criterion
        ``"relative"``: The difference between the last objective function value and the current objective function value relative 
        to the last objective function value is smaller than the stopping_criterion
    :type stopping_method: str
    :param kwargs: The remaining parameters in the child class
    :type kwargs: Dict
    """

    def __init__(self, stopping_criterion, max_line_search, gamma, line_search_method, stopping_method, **kwargs):
        self._stopping_criterion = stopping_criterion
        self._max_line_search = max_line_search
        self._gamma = gamma
        self._line_search_method = line_search_method
        self._stopping_method = stopping_method
        self._obj_fun_value_last = np.inf

    @property
    @abstractmethod
    def obj_fun(self) -> iLQRObjectiveFunction:
        """ Return the objective function for the iLQR algorithm.

        :return: objective function 
        :rtype: iLQRObjectiveFunction
        """
        pass

    @property
    @abstractmethod
    def dynamic_model(self) -> iLQRDynamicModel:
        """ Return the dynamic model function for the iLQR algorithm.

        :return: dynamic model function
        :rtype: iLQRDynamicModel
        """
        pass

    def _vanilla_line_search(self, trajectory, K_matrix, k_vector):
        """The line search method to ensure the value of the objective function is reduced monotonically.

        :param trajectory: Current trajectory
        :type trajectory: array[T, m+n, 1]
        :param K_matrix: Feedback matrix
        :type K_matrix: array[T, m, n]
        :param k_vector: Feedforward vector
        :type k_vector: array[T, m, 1]
        :return: The current_iteration_trajectory after line search and the value of the objective function after the line search
        :rtype: Tuple[array[T, m+n, 1], float]
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros(
            (self.dynamic_model._T, self.dynamic_model._n+self.dynamic_model._m, 1))
        # Line Search if the z value is greater than zero
        for _ in range(self._max_line_search):
            trajectory_current = self.dynamic_model.update_traj(
                trajectory, K_matrix, k_vector, alpha)
            obj_fun_value_current = self.obj_fun.eval_obj_fun(
                trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self._obj_fun_value_last
            alpha = alpha * self._gamma
            if obj_fun_value_delta < 0:
                return trajectory_current, obj_fun_value_current
        return trajectory, self._obj_fun_value_last

    def _feasibility_line_search(self, trajectory, K_matrix, k_vector):
        """To ensure the value of the objective function is reduced monotonically, 
        and ensure the trajectory for the next iteration is feasible.

        :param trajectory: Current trajectory
        :type trajectory: array[T, m+n, 1]
        :param K_matrix: Feedback matrix
        :type K_matrix: array[T, m, n]
        :param k_vector: Feedforward vector
        :type k_vector: array[T, m, 1]
        :return: The current_iteration_trajectory after line search and the value of the objective function after the line search
        :rtype: Tuple[array[T, m+n, 1], float]
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros(
            (self.dynamic_model._T, self.dynamic_model._n+self.dynamic_model._m, 1))
        # Line Search if the z value is greater than zero
        for _ in range(self._max_line_search):
            trajectory_current = self.dynamic_model.update_traj(
                trajectory, K_matrix, k_vector, alpha)
            obj_fun_value_current = self.obj_fun.eval_obj_fun(
                trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self._obj_fun_value_last
            alpha = alpha * self._gamma
            if obj_fun_value_delta < 0 and (not np.isnan(obj_fun_value_delta)):
                return trajectory_current, obj_fun_value_current
        return trajectory, self._obj_fun_value_last

    def _none_line_search(self, trajectory, K_matrix, k_vector):
        """Do not use any line search method.

        :param trajectory: Current trajectory
        :type trajectory: array[T, m+n, 1]
        :param K_matrix: Feedback matrix
        :type K_matrix: array[T, m, n]
        :param k_vector: Feedforward vector
        :type k_vector: array[T, m, 1]
        :return: The current_iteration_trajectory after line search and the value of the objective function after the line search
        :rtype: Tuple[array[T, m+n, 1], float]
        """
        trajectory_current = self.dynamic_model.update_traj(
            trajectory, K_matrix, k_vector, 1)
        obj_fun_value_current = self.obj_fun.eval_obj_fun(trajectory_current)
        return trajectory_current, obj_fun_value_current

    def _vanilla_stopping_criterion(self, obj_fun_value_current):
        """Check the amount of change of the objective function. If the amount of change 
        is less than the specific value, the stopping criterion is satisfied.

        :param obj_fun_value_current: Current objective function value
        :type obj_fun_value_current: float
        :return: Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        :rtype: bool
        """
        obj_fun_value_delta = obj_fun_value_current - self._obj_fun_value_last
        if (abs(obj_fun_value_delta) < self._stopping_criterion):
            return True
        return False

    def _relative_stopping_criterion(self, obj_fun_value_current):
        """ Check the amount of change of the objective function relative to the current objective function value. 
        If the amount of change is less than the specific value, the stopping criterion is satisfied.

        :param obj_fun_value_current: Current objective function value
        :type obj_fun_value_current: float
        :return: Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        :rtype: bool
        """
        obj_fun_value_delta = obj_fun_value_current - self._obj_fun_value_last
        if (abs(obj_fun_value_delta/self._obj_fun_value_last) < self._stopping_criterion):
            return True
        return False

    def forward_pass(self, trajectory, K_matrix, k_vector):
        """Forward pass in the iLQR algorithm.

        :param trajectory: Current trajectory
        :type trajectory: array[T, m+n, 1]
        :param K_matrix: Feedback matrix
        :type K_matrix: array[T, m, n]
        :param k_vector: Feedforward vector
        :type k_vector: array[T, m, 1]
        :return: ``new_trajectory`` The new trajectory. 
            ``C_matrix`` The new Hessian matrix of the objective function.
            ``c_vector`` The new gradient vector of the objective function.
            ``F_matrix`` The new gradient matrix of the dynamic function.
            ``obj_fun_value_current`` The new objective function value with respect to the new trajectory. 
            ``is_stop`` Whether the stopping criterion is satisfied.
        :rtype: Tuple[array[T, m+n, 1], array[T, n+m, n+m], array[T, n+m, 1], array[T, n, n+m], float, bool]
        """
        # Do line search
        if self._line_search_method == "vanilla":
            new_trajectory, obj_fun_value_current = self._vanilla_line_search(
                trajectory, K_matrix, k_vector)
        elif self._line_search_method == "feasibility":
            new_trajectory, obj_fun_value_current = self._feasibility_line_search(
                trajectory, K_matrix, k_vector)
        elif self._line_search_method == "none":
            new_trajectory, obj_fun_value_current = self._none_line_search(
                trajectory, K_matrix, k_vector)
        # Check the stopping criterion
        if self._stopping_method == "vanilla":
            is_stop = self._vanilla_stopping_criterion(obj_fun_value_current)
        elif self._stopping_method == "relative":
            is_stop = self._relative_stopping_criterion(obj_fun_value_current)
        # Do forward pass
        C_matrix = self.obj_fun.eval_hessian_obj_fun(new_trajectory)
        c_vector = self.obj_fun.eval_grad_obj_fun(new_trajectory)
        F_matrix = self.dynamic_model.eval_grad_dynamic_model(new_trajectory)
        # Finally update the objective_function_value_last
        self._obj_fun_value_last = obj_fun_value_current
        return new_trajectory, C_matrix, c_vector, F_matrix, obj_fun_value_current, is_stop

    def backward_pass(self, C_matrix, c_vector, F_matrix):
        """Backward pass in the iLQR algorithm.

        :param C_matrix: Hessian matrix of the objective function
        :type C_matrix: array[T, n+m, n+m]
        :param c_vector: Gradient vector of the objective function
        :type c_vector: array[T, n+m, 1]
        :param F_matrix: Gradient matrix of the dynamic function
        :type F_matrix:  array[T, n, n+m]
        :return: ``feedback matrix K`` and ``feedforward vector k``
        :rtype: Tuple[array[T, m, n],array[T, m, 1]]
        """
        K_matrix, k_vector = self._backward_pass_static(
            self.dynamic_model._m, self.dynamic_model._n, self.dynamic_model._T, C_matrix, c_vector, F_matrix)
        return K_matrix, k_vector

    @staticmethod
    @njit
    def _backward_pass_static(m, n, T, C_matrix, c_vector, F_matrix):
        V_matrix = np.zeros((n, n))
        v_vector = np.zeros((n, 1))
        K_matrix_list = np.zeros((T, m, n))
        k_vector_list = np.zeros((T, m, 1))
        for i in range(T-1, -1, -1):
            Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
            q_vector = c_vector[i] + F_matrix[i].T@v_vector
            Q_uu = Q_matrix[n:n+m, n:n+m].copy()
            Q_ux = Q_matrix[n:n+m, 0:n].copy()
            q_u = q_vector[n:n+m].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu, Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu, q_u)
            V_matrix = Q_matrix[0:n, 0:n] +\
                Q_ux.T@K_matrix_list[i] +\
                K_matrix_list[i].T@Q_ux +\
                K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            v_vector = q_vector[0:n] +\
                Q_ux.T@k_vector_list[i] +\
                K_matrix_list[i].T@q_u +\
                K_matrix_list[i].T@Q_uu@k_vector_list[i]
        return K_matrix_list, k_vector_list

    def get_obj_fun_value(self):
        """Return the current objective function value.

        :return: Current objective function value
        :rtype: float
        """
        return self._obj_fun_value_last

    def set_obj_fun_value(self, obj_fun_value):
        """Set the value of the objective function.

        :param obj_fun_value: New objective function value
        :type obj_fun_value: float
        """
        self._obj_fun_value_last = obj_fun_value

    def set_init_state(self, new_state):
        """Set the init state of the dynamic system

        :param new_state: The new state
        :type new_state: array(n, 1)
        """
        self.dynamic_model._init_state = new_state

    def set_obj_add_param(self, new_add_param):
        """Set the values of the additional parameters in the objective function

        :param new_add_param: The new values to the additioanl variables
        :type new_add_param: array(T, p)
        """
        self.obj_fun._add_param = new_add_param

    def get_obj_add_param(self):
        """Get the values of the additional parameters in the objective function

        :return: additional parameters in the objective function
        :rtype: array(T, p)
        """
        return self.obj_fun._add_param
