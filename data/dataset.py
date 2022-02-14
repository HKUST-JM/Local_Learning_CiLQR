from __future__ import annotations
from typing import Union
import torch
import numpy as np
from .data import Data
from .data import ACCESSIBLE_KEY
from utils.config import global_config
import copy

class Dataset(object):
    """This is a class for building the dataset in the learning process.

    :param buffer_size: The size of the dataset. 
    :type buffer_size: int
    :param state_dim: The dimension of the state data. For example, 
        the state dimension of a binary image 
        is (512, 512). The state of a cartpole system is (4,) or 4, 
        which includes the position, velocity, angle of the
        pole, and the angular velocity of the pole.
    :type state_dim: Tuple(int) or int
    :param action_dim: The dimension of the action data. For example, the 
        state dimension of a cartpole system is 1, 
        which is the force applied to the cart. 
        The action dimension of a vehicle system is 2, 
        which are the steering angle and the accelaration. 
    :type action_dim: int
    """

    def __init__(self, buffer_size,
                 state_dim: Union[Tuple(int), int],
                 action_dim: int):
        if isinstance(state_dim, int):
            state_dim = (state_dim,)
        self._buffer_size = buffer_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._update_key = 0
        self._total_update = 0  # totally number of obtained data

    def update_dataset(self, data: Data):
        """Update the new data into the dataset. If the dataset is full, 
        then this method will remove the oldest data in the dataset,
        and update the new data alternatively.

        :param data: New data
        :type data: Data
        """
        if self._total_update == 0: # If this is the first update, initial the dataset
            self._dataset_dict = {}
            for key in data._data_dict:
                if global_config.is_cuda:
                    if key == "state" or key == "next_state":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size, *data._data_dict[key].shape[1:])).cuda()
                    elif key == "action":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size, data._data_dict[key].shape[1])).cuda()
                    elif key == "reward":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size)).cuda()
                    elif key == "done_flag":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool).cuda()
                else:
                    if key == "state" or key == "next_state":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size, *data._data_dict[key].shape[1:]))
                    elif key == "action":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size, data._data_dict[key].shape[1]))
                    elif key == "reward":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size))
                    elif key == "done_flag":
                        self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool)
        self._total_update += len(data)
        # if not exceed the last data in the dataset
        if self._update_key+len(data) <= self._buffer_size:
            for key in self._dataset_dict:    
                self._dataset_dict[key][self._update_key:self._update_key + len(data)] = data._data_dict[key]
            self._update_key += len(data)
            if self._update_key == self._buffer_size:
                self._update_key = 0
        else:  # if exceed
            exceed_number = len(data) + self._update_key - self._buffer_size
            for key in self._dataset_dict:
                self._dataset_dict[key][self._update_key:] = data._data_dict[key][:self._buffer_size-self._update_key]
                ##########################
                self._dataset_dict[key][:exceed_number] = data._data_dict[key][self._buffer_size-self._update_key:]
            self._update_key = exceed_number

    def get_current_buffer_size(self):
        """Get the current data buffer size. If the number of the current data is less than the buffer size, 
        return current number of data. Otherwise, return the size of the dataset.

        :return: [description]
        :rtype: [type]
        """
        return min(self._total_update, self._buffer_size)

    def fetch_all_data(self):
        """Return all the data in the dataset.

        :return: All data
        :rtype: Data
        """
        index = list(range(self._buffer_size))
        return self.fetch_data_by_index(index)

    def fetch_data_by_index(self, index: list) -> Data:
        """Return the data by specifying the index. For example, if index = [1,2,5], then three datas in the dataset will be returned. 

        :param index: The index of the data
        :type index: list
        :return: Specific data
        :rtype: Data
        """
        temp_dict = {}
        for key in self._dataset_dict:
            temp_dict[key] = self._dataset_dict[key][index]
        data = Data(**temp_dict)
        return data

    def fetch_data_randomly(self, num_of_data: int) -> Data:
        """Return the data with random keyes

        :param num_of_data: How many data will be returned
        :type num_of_data: int
        :return: Data with random keyes
        :rtype: Data
        """
        if self._total_update < self._buffer_size:
            if num_of_data > self._total_update:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._total_update, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._total_update, size=num_of_data, replace=False)
        else:
            if num_of_data > self._buffer_size:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._buffer_size, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._buffer_size, size=num_of_data, replace=False)
        return self.fetch_data_by_index(index)
