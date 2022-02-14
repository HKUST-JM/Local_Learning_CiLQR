import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelBase
from utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class RoboticArmTracking(DynamicModelBase):
    """ In this example, the vehicle packing at 0, 0, heading to the top\\
        We hope the vechile can pack at 0, 0, and head to the right\\
        x0: position_x, x1: position_y, x2: heading anglue, x3: velocity, x4: steering angle, x5: acceleration\\
        If is_with_constraints = True, then the steering angle is limited to [-0.5, 0.5], acceleration is limited to [-2, 2]
    """
    def __init__(self, is_with_constraints = True, T = 120, x = 2, y = 3):
        ##### Dynamic Function ########
        # x0: theta1 
        # x1: theta1 dot 
        # x2: theta2
        # x3: theta2 dot
        # x4: teminal x
        # x5: teminal y
        # x6: tau1
        # x7: tau2
        
        n, m = 6, 2 # number of state = 6, number of action = 2, prediction horizon = 500
        x_u_var = sp.symbols('x_u:8')
        m1 = 1
        m2 = 1
        self.l1 = 1
        self.l2 = 2
        g = 9.8
        h = 0.01 # sampling time
        H = sp.Matrix([
            [((1/3)*m1 + m2)*(self.l1**2),          (1/2)*m2*self.l1*self.l2*sp.cos(x_u_var[0] - x_u_var[2])],
            [(1/2)*m2*self.l1*self.l2*sp.cos(x_u_var[0] - x_u_var[2]),               (1/3)*m2*(self.l2**2)   ]
        ])
        H_inv = H.inv()
        Q = np.asarray([
            [-0.5*m2*self.l1*self.l2*(x_u_var[3]**2)*sp.sin(x_u_var[0] - x_u_var[2]) + 0.5*m1*g*self.l1*sp.sin(x_u_var[0])+m2*g*self.l1*sp.sin(x_u_var[0])],
            [ 0.5*m2*self.l1*self.l2*(x_u_var[1]**2)*sp.sin(x_u_var[0] - x_u_var[2]) + 0.5*m2*g*self.l2*sp.sin(x_u_var[2])]])
        tau =  np.asarray([
            [x_u_var[6]],
            [x_u_var[7]]
            ])
        temp = H_inv@Q + H_inv@tau
        theta1_ddot = temp[0,0]
        theta2_ddot = temp[1,0]
        dynamic_function = sp.Array([  
            x_u_var[0] + h*x_u_var[1],
            x_u_var[1] + h*theta1_ddot,
            x_u_var[2] + h*x_u_var[3],
            x_u_var[3] + h*theta2_ddot,
            self.l1*sp.sin(x_u_var[0]) + self.l2*sp.sin(x_u_var[2]),
            self.l1*sp.cos(x_u_var[0]) + self.l2*sp.cos(x_u_var[2])])
        init_state = np.asarray([0, 0, 0, 0, 0, self.l1+self.l2],dtype=np.float64).reshape(-1,1)
        init_action = np.zeros((T,m,1))
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-100, 100], [-100, 100]]) 
            # other_constr = [-((x_u_var[4] - self._ellip_center[0])**2/(self._ellip_r[0]**2) + (x_u_var[5] - self._ellip_center[1])**2/(self._ellip_r[1]**2) - 1)]
            other_constr = []
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        position_var = sp.symbols("p:2") # x and y
        # add_param_obj = np.hstack([ np.vstack([-3*np.ones((int(int(T/4)), 1)), 0*np.ones((int(int(T/4)), 1)), 3*np.ones((int(int(T/4)), 1)), 0*np.ones((int(int(T/4)), 1))]),
        #                             np.vstack([0*np.sin(54/180)*np.ones((int(int(T/4)), 1)), -3*np.ones((int(int(T/4)), 1)), 0*np.ones((int(int(T/4)), 1)), 3*np.ones((int(int(T/4)), 1))])])
        add_param_obj = np.hstack([ 
            np.vstack([
                np.linspace(0, -2, int(T/6)).reshape(-1,1), 
                np.linspace(-2, -2, int(T/6)).reshape(-1,1), 
                np.linspace(-2, 0, int(T/6)).reshape(-1,1), 
                np.linspace(0, 2, int(T/6)).reshape(-1,1),  
                np.linspace(2, 2, int(T/6)).reshape(-1,1),  
                np.linspace(2, 0, int(T/6)).reshape(-1,1)]),
            np.vstack([
                np.linspace(3, 2, int(T/6)).reshape(-1,1),  
                np.linspace(2, 2, int(T/6)).reshape(-1,1),  
                np.linspace(2, 3, int(T/6)).reshape(-1,1), 
                np.linspace(3, 2, int(T/6)).reshape(-1,1),
                np.linspace(2, 2, int(T/6)).reshape(-1,1), 
                np.linspace(2, 3, int(T/6)).reshape(-1,1)])
            ])
        C_matrix =    np.diag([0.,       0.,     0.,         0.,          1.,                         1.,                           0.,              0.])
        r_vector = np.asarray([0.,       0.,     0.,         0.,          position_var[0],            position_var[1],              0.,              0.])
        runing_obj = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector) 
        obj_fun = runing_obj
        
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param_obj)

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(4, 3), xlim=(-4,4), ylim=(-2,4), is_equal = False)
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        
        pole1 = patches.FancyBboxPatch((0, 0), 0.2, self.l1+0.1, "round,pad=0.1")
        pole1.set_color('C0')
        pole1_ = patches.FancyBboxPatch((0, 0), 0.2, self.l1+0.1, "round,pad=0.12")
        pole1_.set_color('black')
        pole2 = patches.FancyBboxPatch((0, 0), 0.2, self.l2+0.1, "round,pad=0.1")
        pole2.set_color('C0')
        pole2_ = patches.FancyBboxPatch((0, 0), 0.2, self.l2+0.1, "round,pad=0.12")
        pole2_.set_color('black')
        joint1 = patches.Circle((0, 0), 0.05)
        joint1.set_color('black')
        joint2 = patches.Circle((0, 0), 0.05)
        joint2.set_color('black')
        joint3 = patches.Circle((0, 0), 0.3)
        joint3.set_color((0.9290, 0.6940, 0.1250))
        joint3_ = patches.Circle((0, 0), 0.34)
        joint3_.set_color('black')
        # obs = patches.Ellipse(self._ellip_center, 2*self._ellip_r[0], 2*self._ellip_r[1])
        # obs.set_color('grey')

        base = patches.Polygon(np.asarray([[-0.5,-0.5],[-0.75,-1],[0.75,-1],[0.5,-0.5]]))
        base.set_color('silver')
        base.set_ec('black')
        base2 = patches.FancyBboxPatch((-0.1, -0.5), 0.2, 0.5, "round,pad=0.1")
        base2.set_color('silver')
        base2.set_ec('black')
        ax.add_patch(base2)
        ax.add_patch(base)
        ax.add_patch(pole1_)
        ax.add_patch(pole1)
        ax.add_patch(pole2_)
        ax.add_patch(pole2)
        ax.add_patch(joint1)
        ax.add_patch(joint2)
        ax.add_patch(joint3)
        plt.plot(trajectory[:, 4], trajectory[:, 5], 'C1')
        # ax.add_patch(obs)
        self._is_interrupted=False
        for i in range(self.T):
            self.play_trajectory_current = trajectory[i,:,0]
            # draw pole1
            t_start = ax.transData
            x1 = -0.1*np.cos(self.play_trajectory_current[0])
            y1 =  0.1*np.sin(self.play_trajectory_current[0])
            rotate_center = t_start.transform([x1, y1])
            pole1.set_x(x1)
            pole1.set_y(y1)
            pole1_.set_x(x1)
            pole1_.set_y(y1)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[0])
            t_end = t_start + t
            pole1.set_transform(t_end)
            pole1_.set_transform(t_end)
            # draw pole2
            x2 = self.l1*np.sin(self.play_trajectory_current[0]) - 0.1*np.cos(self.play_trajectory_current[2])
            y2 = self.l1*np.cos(self.play_trajectory_current[0]) + 0.1*np.sin(self.play_trajectory_current[2])
            rotate_center = t_start.transform([x2, y2])
            pole2.set_x(x2)
            pole2.set_y(y2)
            pole2_.set_x(x2)
            pole2_.set_y(y2)
            joint2.center = (self.l1*np.sin(self.play_trajectory_current[0]), self.l1*np.cos(self.play_trajectory_current[0]))
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[2])
            t_end = t_start + t
            pole2.set_transform(t_end)
            pole2_.set_transform(t_end)

            x3 = self.l1*np.sin(self.play_trajectory_current[0]) + self.l2*np.sin(self.play_trajectory_current[2])
            y3 = self.l1*np.cos(self.play_trajectory_current[0]) + self.l2*np.cos(self.play_trajectory_current[2])
            joint3.center=(x3,y3)
            joint3_.center=(x3,y3)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True
