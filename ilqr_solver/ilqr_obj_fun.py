import numpy as np
import sympy as sp
from numba import njit

class iLQRObjectiveFunction(object):
    """This is a wrapper class for the objective function. There are two kinds of variables in the objective function. 
    The first kind is the state and action variables. The second kind is the additional variables.
    The gradient and hessian will obly be compuated with respect to the first find of variables.
    The additional variable are used to design the time variant objective functions. For example, for the objective function 
    with the terminal penality, the weighing parameters are changed in the last time stamp. To be more specific, if a linear
    quadratic objective function is considered, :math:`J=(x-r)^TQ(x-r)`, then we can define :math:`Q` as time variant. In the 
    ``cart_pole_swingup2`` example, the following codes are used:
    ::
       C_matrix_diag = sp.symbols("c:6")
       r_vector = np.asarray([0, 0, 0, 1, 0, 0])
       add_param_obj = np.zeros((T, 6), dtype = np.float64)
       for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((0.1, 0.1, 1, 1, 0.1, 1))
            else: 
                add_param_obj[tau] = np.asarray((0.1, 0.1, 10000, 10000, 1000, 0))
       obj_fun = (x_u_var- r_vector)@np.diag(np.asarray(C_matrix_diag))@(x_u_var- r_vector)

    Also, in the tracking problem with a time variant reference, the additional parameter also can be used. In the example of 
    ``two_link_planar_manipulator``, the reference is different in each iLQR optimization, therefore we use the additional parameters.
    ::
       position_var = sp.symbols("p:2") # x and y
       C_matrix =    np.diag([0.,      10.,     0.,        10.,          10000,                             10000,                 1,           1])
       r_vector = np.asarray([0.,       0.,     0.,         0.,          position_var[0],            position_var[1],              0.,          0.])
       obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector) 

    If the additional parameters are not used, leave them to be None. 

    :param obj_fun: The function of the objetive function.
    :type obj_fun: sympy symbolic expression
    :param x_u_var: State and action variables in the objective function
    :type x_u_var: (sympy.symbol, sympy.symbol, ...) 
    :param add_param_var: Introduce the additional variables that are not derived, defaults to None
    :type add_param_var: (sympy.symbol, sympy.symbol, ...), optional
    :param add_param: Give the values to the additioanl variables (totally p variables), defaults to None
    :type add_param: array(T, p), optional
    """
    def __init__(self, obj_fun, x_u_var, add_param_var = None, add_param = None): 
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
        self._obj_fun_lamdify = njit(sp.lambdify([x_u_var,add_param_var], obj_fun, "numpy"))
        gradient_objective_function_array = sp.derive_by_array(obj_fun, x_u_var)
        self._grad_obj_fun_lamdify = njit(sp.lambdify([x_u_var, add_param_var], gradient_objective_function_array,"numpy"))       
        hessian_objective_function_array = sp.derive_by_array(gradient_objective_function_array, x_u_var)
        # A stupid method to ensure each element in the hessian matrix is in the type of float64
        self._hessian_obj_fun_lamdify = njit(sp.lambdify([x_u_var, add_param_var], np.asarray(hessian_objective_function_array)+1e-100*np.eye(hessian_objective_function_array.shape[0]),"numpy"))
        self._add_param = add_param

    def eval_obj_fun(self, trajectory):
        """Given the trajectory of the state and action variables, evaluate the objective function value.

        :param trajectory: Trajectory of the state and action variables.
        :type trajectory: array(T, m+n, 1) 
        :return: Objective function value.
        :rtype: double
        """
        return self._eval_obj_fun_static(self._obj_fun_lamdify, trajectory, self._add_param)

    def eval_grad_obj_fun(self, trajectory):
        """Given the trajectory of the state and action variables, 
        evaluate gradient of the objective function with respect to the state and action variables.

        :param trajectory: Trajectory of the state and action variables.
        :type trajectory: array(T, m+n, 1) 
        :return: Jacobian matrix of the objective function
        :rtype: array(T, m+n,1) 
        """
        return self._eval_grad_obj_fun_static(self._grad_obj_fun_lamdify, trajectory, self._add_param)

    def eval_hessian_obj_fun(self, trajectory):
        """Given the trajectory of the state and action variables, 
        evaluate second-order derivative of the objective function with respect to the state and action variables.

        :param trajectory: Trajectory of the state and action variables.
        :type trajectory: array(T, m+n, 1) 
        :return: Hessian matrix of the objective function
        :rtype: array(T, m+n, m+n) 
        """
        return self._eval_hessian_obj_fun_static(self._hessian_obj_fun_lamdify, trajectory, self._add_param)

    @staticmethod
    @njit
    def _eval_obj_fun_static(obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        if add_param is None:
            add_param = np.zeros((T,1))
        obj_value = 0.
        for tau in range(T):
            obj_value = obj_value + np.asarray(obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return obj_value
    
    @staticmethod
    @njit
    def _eval_grad_obj_fun_static(grad_obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        m_n = int(trajectory.shape[1])
        if add_param is None:
            add_param = np.zeros((T,1))
        grad_all_tau = np.zeros((T, m_n, 1))
        for tau in range(T):
            grad_all_tau[tau] = np.asarray(grad_obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64).reshape(-1,1)
        return grad_all_tau
    
    @staticmethod
    @njit
    def _eval_hessian_obj_fun_static(hessian_obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        m_n = int(trajectory.shape[1])
        if add_param is None:
            add_param = np.zeros((T,1))
        hessian_all_tau = np.zeros((T, m_n, m_n))
        for tau in range(T):
            hessian_all_tau[tau] = np.asarray(hessian_obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return hessian_all_tau
