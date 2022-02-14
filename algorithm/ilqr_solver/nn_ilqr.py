from __future__ import annotations
import numpy as np
import time as tm
from scipy import io
import os
import torch
import torch_optimizer as optim
from torch import Tensor, nn
from .basic_ilqr import iLQRWrapper
from utils.Logger import logger
from data import Data, Dataset
from scipy.ndimage import gaussian_filter1d
from .ilqr_dynamic_model import iLQRDynamicModel
from .ilqr_obj_fun import iLQRObjectiveFunction
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data import Data
    from data import Dataset
    from scenario.dynamic_model.dynamic_model import DynamicModelWrapper
VALI_DATASET_SIZE = 10

class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, action_channels, output_channels, is_shorcut=True):
        super().__init__()
        self.is_shorcut = is_shorcut
        self.bn1 = nn.BatchNorm1d(action_channels)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(action_channels, output_channels)

        self.bn2 = nn.BatchNorm1d(output_channels)
        self.linear2 = nn.Linear(output_channels, output_channels)

        self.shorcut = nn.Linear(action_channels, output_channels)
        self.main_track = nn.Sequential(
            self.bn1, self.relu, self.linear1, self.bn2,  self.relu, self.linear2)

    def forward(self, X):
        if self.is_shorcut:
            Y = self.main_track(X) + self.shorcut(X)
        else:
            Y = self.main_track(X) + X
        return torch.nn.functional.relu(Y)


class SmallResidualNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no = 64
        layer2_no = 32
        layer3_no = 16
        layer4_no = 8
        self.layer = nn.Sequential(Residual(in_dim, layer1_no),
                                   Residual(layer1_no, layer2_no),
                                   Residual(layer2_no, layer3_no),
                                   Residual(layer3_no, layer4_no),
                                   nn.Linear(layer4_no, out_dim))

    def forward(self, x):
        x = self.layer(x)
        return x


class SmallNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no = 128
        layer2_no = 64
        self.layer = nn.Sequential(
            nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(),
            nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(
                layer2_no), nn.ReLU(),
            nn.Linear(layer2_no, out_dim))

    def forward(self, x):
        x = self.layer(x)
        return x

class LargeNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no = 800
        layer2_no = 400
        self.layer = nn.Sequential(
            nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(),
            nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(
                layer2_no), nn.ReLU(),
            nn.Linear(layer2_no, out_dim))

    def forward(self, x):
        x = self.layer(x)
        return x


class NNiLQRDynamicModel(iLQRDynamicModel):
    """ NNiLQRDynamicModel uses a neural network to fit the dynamic model of a system. This algorithm can only be implemented on a cuda device.
    """
    def __init__(self, network, init_state, init_action):
        """ Initialization
            network : nn.module
                Networks used to train the system dynamic model
            init_state : array(n,1)
                Initial system state
            init_action_traj : array(T, m, 1)
                Initial action trajectory used to generalize the initial trajectory
        """
        self._init_action = init_action
        self._init_state = init_state
        self._n = init_state.shape[0]
        self._m = init_action.shape[1]
        self._T = int(init_action.shape[0])
        self._model = network.cuda()
        self._F_matrix = torch.zeros(
            (self._T, self._n, self._m+self._n)).cuda()
        self.__constant1 = torch.eye(self._n).cuda()

    def _process_data(self, dataset) -> Tuple[Tensor, Tensor]:
        data = dataset.fetch_all_data()
        traj = data.state
        X = traj[[n for n in range(len(traj)) if ((n+1) % self._T != 0)]]
        Y = traj[[n+1 for n in range(len(traj))
                  if ((n+1) % self._T != 0)], 0:self._n]
        return X, Y

    def pretrain(self, dataset_train: Dataset, dataset_vali: Dataset, max_epoch=50000, stopping_criterion=1e-4, lr=1e-3, model_name="NeuralDynamic.model"):
        """ Pre-train the model by using randomly generalized data

            Parameters
            ------------
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            dataset_vali : DynamicModelDataSetWrapper
                Data set for validation
            max_epoch : int
                Maximum number of epochs if stopping criterion is not reached
            stopping_criterion : double
                If the objective function of the training set is less than 
                the stopping criterion, the training is stopped
            lr : double
                Learning rate
            model_name : string
                When the stopping criterion, 
                the model with the given name will be saved as a file
        """
        # if the model exists, load the model directly
        model_path = logger.logger_path
        if model_path is None:
            raise Exception(
                "Please call logger.set_folder_name(folder_name).set_is_use_logger(True) to enable the logger function!")
        result_train_loss = np.zeros(max_epoch)
        result_vali_loss = np.zeros(int(max_epoch/100))
        if not os.path.exists(os.path.join(model_path, model_name)):
            logger.info("[+ +] Model file \"" + model_name +
                        "\" does NOT exist. Pre-traning starts...")
            loss_fun = nn.MSELoss()
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            optimizer = optim.RAdam(
                self._model.parameters(), lr=lr, weight_decay=1e-4)
            X_train, Y_train = self._process_data(dataset_train)
            X_vali, Y_vali = self._process_data(dataset_vali)
            time_start_pretraining = tm.time()
            for epoch in range(max_epoch):
                #### Training ###
                self._model.train()
                optimizer.zero_grad()
                Y_prediction = self._model(X_train)
                obj_train = loss_fun(Y_prediction, Y_train)
                obj_train.backward()
                optimizer.step()
                result_train_loss[epoch] = obj_train.item()
                if obj_train.item() < stopping_criterion or epoch % 100 == 0:  # Check stopping criterion
                    ## Evaluation ###
                    self._model.eval()
                    # Forward Propagation
                    Y_prediction = self._model(X_vali)
                    obj_vali = loss_fun(Y_prediction, Y_vali)
                    ##### Print #####
                    logger.info("[+ +] Epoch: %5d     Train Loss: %.5e     Vali Loss:%.5e" % (
                        epoch + 1,    obj_train.item(),    obj_vali.item()))
                    result_vali_loss[int(np.ceil(epoch/100))] = obj_vali.item()
                    if obj_train.item() < stopping_criterion:
                        time_end_preraining = tm.time()
                        time_pretraining = time_end_preraining - time_start_pretraining
                        logger.info(
                            "[+ +] Pretraining finished! Model file \"" + model_name + "\" is saved!")
                        logger.info("[+ +] Pretraining time: %.8f" %
                                    (time_pretraining))
                        torch.save(self._model.state_dict(),
                                   os.path.join(model_path, model_name))
                        io.savemat(os.path.join(model_path, model_name + "_training.mat"),
                                   {"Train_loss": result_train_loss, "Vali_loss": result_vali_loss})
                        return
            raise Exception("Maximum epoch in the pretraining is reached!")
        else:
            logger.info("[+ +] Model file \"" +
                        model_name + "\" exists. Loading....")
            self._model.load_state_dict(torch.load(
                os.path.join(model_path, model_name)))
            self._model.eval()

    def retrain(self, dataset, max_epoch=10000, stopping_criterion=1e-3, lr=1e-3):
        logger.info("[+ +] Re-traning starts...")
        loss_fun = nn.MSELoss()
        # optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.RAdam(self._model.parameters(),
                                lr=lr, weight_decay=1e-4)
        X_train, Y_train = self._process_data(dataset)
        for epoch in range(max_epoch):
            #### Training ###
            self._model.train()
            optimizer.zero_grad()
            Y_prediction = self._model(X_train)
            obj_train = loss_fun(Y_prediction, Y_train)
            obj_train.backward()
            optimizer.step()
            if obj_train.item() < stopping_criterion or epoch % 100 == 0:  # Check stopping criterion
                logger.info("[+ +] Epoch: %5d   Train Obj: %.5e" % (
                    epoch + 1,     obj_train.item()))
                if obj_train.item() < stopping_criterion:
                    logger.info("[+ +] Re-training finished!")
                    self._model.eval()
                    return
        raise Exception("Maximum epoch in the retraining is reached!")

    def eval_traj(self, init_state=None, action_traj=None):
        if init_state is None:
            init_state = self._init_state
        if action_traj is None:
            action_traj = self._init_action
        action_traj_cuda = torch.from_numpy(action_traj).float().cuda()
        trajectory = torch.zeros(self._T, self._n+self._m).cuda()
        trajectory[0] = torch.from_numpy(
            np.vstack((init_state, action_traj[0]))).float().cuda().reshape(-1)
        with torch.no_grad():
            for tau in range(self._T-1):
                trajectory[tau+1,
                           :self._n] = self._model(trajectory[tau, :].reshape(1, -1))
                trajectory[tau+1, self._n:] = action_traj_cuda[tau+1, 0]
        return trajectory.cpu().double().numpy().reshape(self._T, self._m+self._n, 1)

    def update_traj(self, old_traj, K_matrix, k_vector, alpha):
        new_traj = np.zeros((self._T, self._m+self._n, 1))
        new_traj[0] = old_traj[0]  # initial states are the same
        for tau in range(self._T-1):
            # The amount of change of state x
            delta_x = new_traj[tau, :self._n] - old_traj[tau, :self._n]
            # The amount of change of action u
            delta_u = K_matrix[tau]@delta_x+alpha*k_vector[tau]
            # The real action of next iteration
            action_u = old_traj[tau, self._n:self._n+self._m] + delta_u
            new_traj[tau, self._n:] = action_u
            with torch.no_grad():
                new_traj[tau+1, 0:self._n, 0] = self._model(torch.from_numpy(
                    new_traj[tau, :].T).float().cuda()).cpu().double().numpy()
            # dont care the action at the last time stamp, because it is always zero
        return new_traj

    def eval_grad_dynamic_model(self, trajectory):
        trajectory_cuda = torch.from_numpy(trajectory[:, :, 0]).float().cuda()
        for tau in range(0, self._T):
            x = trajectory_cuda[tau]
            x = x.repeat(self._n, 1)
            x.requires_grad_(True)
            y = self._model(x)
            y.backward(self.__constant1)
            self._F_matrix[tau] = x.grad.data
        return self._F_matrix.cpu().double().numpy()


class NNiLQR(iLQRWrapper):
    """This is a Neural Network iLQR class
    """

    def __init__(self,
                 iLQR_stopping_criterion=1e-6,
                 max_line_search=10,
                 gamma=0.5,
                 line_search_method="vanilla",
                 stopping_method="vanilla",
                 network_class=LargeNetwork,
                 trial_no=100,
                 training_stopping_criterion=1e-4,
                 iLQR_max_iter=1000,
                 decay_rate=0.98,
                 decay_rate_max_iters=200,
                 gaussian_filter_sigma=5,
                 gaussian_noise_sigma=1):
        super().__init__(stopping_criterion=iLQR_stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method,
                         network_class=network_class,
                         trial_no=trial_no,
                         training_stopping_criterion=training_stopping_criterion,
                         iLQR_max_iter=iLQR_max_iter,
                         decay_rate=decay_rate,
                         decay_rate_max_iters=decay_rate_max_iters,
                         gaussian_filter_sigma=gaussian_filter_sigma,
                         gaussian_noise_sigma=gaussian_noise_sigma)
        self._network_class = network_class
        self._trial_no = trial_no
        self._training_stopping_criterion = training_stopping_criterion
        self._iLQR_max_iter = iLQR_max_iter
        self._decay_rate = decay_rate
        self._decay_rate_max_iters = decay_rate_max_iters
        self._gaussian_filter_sigma = gaussian_filter_sigma
        self._gaussian_noise_sigma = gaussian_noise_sigma

    def init(self, scenario: DynamicModelWrapper) -> BasiciLQR:

        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.name +
                            "\" cannot learn with LogBarrieriLQR")
        # Initialize the dynamic_model and objective function
        self._example_name = scenario.name
        self._dynamic_model = iLQRDynamicModel(dynamic_function=scenario.dynamic_function,
                                               x_u_var=scenario.x_u_var,
                                               box_constr=scenario.box_constr,
                                               init_state=scenario.init_state,
                                               init_action=scenario.init_action,
                                               add_param_var=None,
                                               add_param=None)
        self._obj_fun = iLQRObjectiveFunction(obj_fun=scenario.obj_fun,
                                              x_u_var=scenario.x_u_var,
                                              add_param_var=scenario.add_param_var,
                                              add_param=scenario.add_param)
        network = self._network_class(scenario.n + scenario.m, scenario.n)
        action_constr = scenario.box_constr[scenario.n:]
        if np.any(np.isinf(action_constr)):
            raise Exception(
                "The constraint of dynamic system action must not be inf!")

        def generate_random_trajectory():
            actions = np.expand_dims(np.random.uniform(action_constr[:, 0], action_constr[:, 1], size=[
                                     scenario.T, len(action_constr[:, 0])]), axis=2)
            traj = self._dynamic_model.eval_traj(action_traj=actions)
            new_data = Data(state=traj[:, :,  0])  # all data is saved in state
            return new_data
        self._dataset_train = Dataset(
            buffer_size=self._trial_no*scenario.T, state_dim=scenario.n + scenario.m, action_dim=1)
        for _ in range(self._trial_no):
            self._dataset_train.update_dataset(generate_random_trajectory())
        dataset_vali = Dataset(
            buffer_size=VALI_DATASET_SIZE*scenario.T, state_dim=scenario.n + scenario.m, action_dim=1)
        for _ in range(VALI_DATASET_SIZE):
            dataset_vali.update_dataset(generate_random_trajectory())
        self._nn_dynamic_model = NNiLQRDynamicModel(
            network, scenario.init_state, scenario.init_action)
        self._nn_dynamic_model.pretrain(
            self._dataset_train, dataset_vali, stopping_criterion=self._training_stopping_criterion)
        return self

    def solve(self):
        trajectory = self._dynamic_model.eval_traj()  # init feasible trajectory
        init_obj = self._obj_fun.eval_obj_fun(trajectory)
        logger.info("[+ +] Initial Obj.Val.: %.5e" %
                    (init_obj))
        self.set_obj_fun_value(init_obj)
        new_data = []
        result_obj_val = np.zeros(self._iLQR_max_iter)
        result_iter_time = np.zeros(self._iLQR_max_iter)
        re_train_stopping_criterion = self._training_stopping_criterion
        for i in range(int(self._iLQR_max_iter)):
            if i == 1:  # skip the compiling time
                start_time = tm.time()
            iter_start_time = tm.time()
            C_matrix = self._obj_fun.eval_hessian_obj_fun(trajectory)
            c_vector = self._obj_fun.eval_grad_obj_fun(trajectory)
            F_matrix = self._nn_dynamic_model.eval_grad_dynamic_model(
                trajectory)
            F_matrix = gaussian_filter1d(
                F_matrix, sigma=self._gaussian_filter_sigma, axis=0)
            K_matrix, k_vector = self.backward_pass(
                C_matrix, c_vector, F_matrix)
            trajectory, C_matrix, c_vector, F_matrix, obj_val, isStop = self.forward_pass(
                trajectory, K_matrix, k_vector)
            if i < self._decay_rate_max_iters:
                re_train_stopping_criterion = re_train_stopping_criterion * self._decay_rate
            iter_end_time = tm.time()
            iter_time = iter_end_time-iter_start_time
            logger.info("[+ +] Iter.No.:%3d  Iter.Time:%.3e   Obj.Val.:%.5e" % (
                i,               iter_time,       obj_val,))
            result_obj_val[i] = obj_val
            result_iter_time[i] = iter_time
            if isStop:
                if len(new_data) != 0:  # Ensure the optimal trajectroy being in the dataset
                    trajectory_noisy = trajectory
                else:
                    trajectory_noisy = self.dynamic_model.eval_traj(action_traj=(
                        trajectory[:, self.dynamic_model.n:]+np.random.normal(0, self._gaussian_noise_sigma, [self.dynamic_model.T, self.dynamic_model.m, 1])))
                new_data += [trajectory_noisy]
                data = np.concatenate(
                    new_data[-int(self._trial_no/5):])[:, :, 0]
                self._dataset_train.update_dataset(Data(state=data))
                logger.save_to_json(trajectory=trajectory.tolist(), trajectroy_noisy = trajectory_noisy.tolist())
                self._nn_dynamic_model.retrain(
                    self._dataset_train, max_epoch=100000, stopping_criterion=re_train_stopping_criterion)
                new_data = []
            else:
                new_data += [trajectory]
        end_time = tm.time()
        io.savemat(os.path.join(logger.logger_path,  "_result.mat"), {
                   "obj_val": result_obj_val, "iter_time": result_iter_time})
        logger.info("[+ +] Completed! All Time:%.5e" % (end_time-start_time))

    @property
    def obj_fun(self) -> iLQRObjectiveFunction:
        return self._obj_fun

    @property
    def dynamic_model(self) -> iLQRDynamicModel:
        return self._dynamic_model
