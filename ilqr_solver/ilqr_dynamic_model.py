import numpy as np
import sympy as sp
from numba import njit
class iLQRDynamicModel(object):
    """This class is used to create a dynamic model for solving the iLQR problems. The basic methods provided by this class
    are evaluating the trajectory, calculating the gradient matrix of the dynamic function, and update the trajectroy by 
    providing the feedback matrix K and feedforward vector k in the iLQR algorithm.

    :param dynamic_function: The model dynamic function defined by sympy symbolic array, given by the dynamic model scenario class.
    :type dynamic_function: sympy symbolic array
    :param x_u_var: State and action variables in the model
    :type x_u_var: Tuple[sympy.symbol, sympy.symbol, ...]
    :param constr: The box constraint for each state and action variable in the dynamic system. If no constrained, the corresponding 
        range should be [-np.inf, np.inf]
    :type constr: List[n+m, 2]
    :param init_state: The initial state vector of the system
    :type init_state: array[n, 1]
    :param init_action: The initial action vector to create the initial feasible trajectory
    :type init_action: array[T, m, 1] 
    :param add_param_var: Introduce the additional variables (total number is q) that are not under derivation, defaults to None
    :type add_param_var: Tuple[sympy.symbol, sympy.symbol, ...], optional
    :param add_param: Give the values to the additioanl variables, defaults to None
    :type add_param: array[T, q], optional
    """
    def __init__(self, dynamic_function, x_u_var, box_constr, init_state, init_action, add_param_var = None, add_param = None):
        self._init_state = init_state
        self._init_action = init_action
        self._n = int(init_state.shape[0])
        self._m = int(len(x_u_var) - self._n)
        self._T = int(init_action.shape[0])
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
        self._dynamic_function_lamdify = njit(sp.lambdify([x_u_var, add_param_var], dynamic_function, "math"))
        grad_dynamic_function = sp.transpose(sp.derive_by_array(dynamic_function, x_u_var))
        self._grad_dynamic_function_lamdify = njit(sp.lambdify([x_u_var, add_param_var], grad_dynamic_function, "math"))
        self._add_param = add_param
        self._box_constr = box_constr

    def eval_traj(self, init_state = None, action_traj = None):
        """Evaluate the system trajectory by given initial states and action vector. If init_state = None, action_traj = None,
        then the generated trajectory is based on the initial states and action vector given in the initialization. 

        :param init_state: The initial state used to evaluate trajectory, defaults to None
        :type init_state: array[n, 1], optional
        :param action_traj: The action trajectory used to evaluate trajectory, defaults to None
        :type action_traj: array[T, m, 1], optional
        :return: The generated trajectory
        :rtype: array[T, m+n, 1]
        """
        if init_state is None:
            init_state = self._init_state
        if action_traj is None:
            action_traj = self._init_action
        return self._eval_traj_static(self._dynamic_function_lamdify, init_state, action_traj, self._add_param, self._m, self._n, self._box_constr)

    def update_traj(self, old_traj, K_matrix_all, k_vector_all, alpha): 
        """Generate the new system trajectory by given old trajectory, feedback matrix K, feedforward vector k,
        calculated by the iLQR algorithm and the step size alpha.

        :param old_traj: Trajectory in the last iteration
        :type old_traj: array[T, m+n, 1]
        :param K_matrix_all: Feedback matrix K obtained by iLQR
        :type K_matrix_all: array[T, n, m+n]
        :param k_vector_all: Feedforward vector k obtained by iLQR
        :type k_vector_all: array[T, m+n, 1]
        :param alpha: Step size in this iteration
        :type alpha: float
        :return: Updated trajectory
        :rtype: array[T, m+n, 1]
        """
        return self._update_traj_static(self._dynamic_function_lamdify, self._m, self._n, old_traj, K_matrix_all, k_vector_all, alpha, self._add_param, self._box_constr)

    def eval_grad_dynamic_model(self, trajectory):
        """Evaluate the matrix of the gradient of the dynamic_model given a specific trajectory

        :param trajectory: Specific trajectory
        :type trajectory: array[T, m+n, 1]
        :return: Gradient matrix of the dynamic_model
        :rtype:  array[T, n, m+n]
        """
        return self._eval_grad_dynamic_model_static(self._grad_dynamic_function_lamdify, trajectory, self._add_param)

    @property
    def T(self):
        return self._T
    
    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n


    @staticmethod
    @njit
    def _eval_traj_static(dynamic_model_lamdify, init_state, action_traj, add_param, m, n, constr):
        T = int(action_traj.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        trajectory = np.zeros((T, m+n, 1))
        trajectory[0] = np.vstack((init_state, action_traj[0]))
        for tau in range(T-1):
            trajectory[tau+1, :n, 0] = np.asarray(dynamic_model_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
            trajectory[tau+1, n:] = action_traj[tau+1]
            for i, c in enumerate(constr):
                trajectory[tau, i, 0] = min(max(c[0], trajectory[tau, i, 0]), c[1]) 
        return trajectory

    @staticmethod
    @njit
    def _update_traj_static(dynamic_model_lamdify, m, n, old_traj, K_matrix_all, k_vector_all, alpha, add_param, constr):
        T = int(K_matrix_all.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        new_trajectory = np.zeros((T, m+n, 1))
        new_trajectory[0] = old_traj[0] # initial states are the same
        for tau in range(T-1):
            # The amount of change of state x
            delta_x = new_trajectory[tau, 0:n] - old_traj[tau, 0:n]
            # The amount of change of action u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real action of next iteration
            action_u = old_traj[tau, n:n+m] + delta_u
            new_trajectory[tau,n:] = action_u
            for i, c in enumerate(constr[n:]):
                new_trajectory[tau, n+i, 0] = min(max(c[0], new_trajectory[tau, n+i, 0]), c[1]) 
            new_trajectory[tau+1,:n] = np.asarray(dynamic_model_lamdify(new_trajectory[tau,:,0], add_param[tau]),dtype=np.float64).reshape(-1,1)
            for i, c in enumerate(constr[:n]):
                new_trajectory[tau+1, i, 0] = min(max(c[0], new_trajectory[tau+1, i, 0]), c[1]) 
            
            # dont care the action at the last time stamp, because it is always zero
        return new_trajectory

    @staticmethod
    @njit
    def _eval_grad_dynamic_model_static(grad_dynamic_model_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        F_matrix_initial =  grad_dynamic_model_lamdify(trajectory[0,:,0], add_param[0])
        F_matrix_all = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_all[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T):
            F_matrix_all[tau] = np.asarray(grad_dynamic_model_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return F_matrix_all

