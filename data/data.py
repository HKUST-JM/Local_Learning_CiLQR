from __future__ import annotations
import torch
from torch import Tensor
from typing import Union
from utils.config import global_config
import copy
ACCESSIBLE_KEY = {'state', 'action', 'next_state', 'reward', 'done_flag'}


class Data(object):
    """This is the class for building the reinforcement learning data inlcuding state, action, next_state, reward, and done_flag.
    This is the basic data class for all the algorithms in the ``CuriousRL`` package. All data commnunication in the algorithms are based
    on this class. Each ``Data`` instance can contain none or one or several state(s), action(s), next state(s), reward(s), and done flag(s).

    There are two ways to initial a ``Data`` class. The first way initializes the ``Data`` class from a single data. 
    The Second way initializes the ``Data`` class from multiple data. In the single data mode, the dimension of the action is 1, but
    in the multiple data mode, the dimension of the action is 2. The first dimension is the index of data but the second dimension is the 
    action vector. 

    In terms of the single data, 
    the state and action are given directly without the index of data. For example,
    if an state is a grey image, the type of state is ``Tensor[512,512]``, the type of action is ``Tensor[5]``,
    the type of next state is ``Tensor[512,512]``,
    the type of reward is ``float``, and the type of done flag is ``bool``.  

    In terms of the multiple data, the types of state, action, next_state, 
    reward and done_flag are all ``Tensor`` with the index dimension as the first dimension.  For example,
    if the state is a grey image, then the type of state is ``Tensor[10,512,512]``, the type of action is ``Tensor[10,5]``,
    the type of next state is ``Tensor[10,512,512]``,
    the type of reward is ``Tensor[10,1]``, and the type of done_flag is ``Tensor[10,1]``. In this mode, one ``Data`` instance can contain many pieces of data. 

    .. note::
        ``Numpy.ndarray`` is also supported in this class, which can be used as the alternative type of ``Tensor``.

    .. note::
        State, action, next_state, reward, done_flag should not be given if it is not necessary for the algorithm.

    .. note::
        If action is not given, then the multiple data mode is implemented.

    :param state: State
    :type state: Union[Tensor, numpy.ndarray]
    :param action: Action
    :type action: Union[Tensor, numpy.ndarray]
    :param next_state: Next state
    :type next_state: Union[Tensor, numpy.ndarray]
    :param reward: Reward
    :type reward: Union[Tensor, numpy.ndarray]
    :param done_flag: The flag deciding whether one episode is done
    :type done_flag: Union[Tensor, numpy.ndarray, bool]
    """

    def __init__(self, **kwargs):
        # if numpy, to tensor.
        self._data_dict = {}
        for key in kwargs:
            if key not in ACCESSIBLE_KEY:
                raise Exception(
                    "\"" + key + "\" is not an accessible key in data!")
            if isinstance(kwargs[key], Tensor):
                self._data_dict[key] = kwargs[key]
            else:
                if global_config.is_cuda:
                    self._data_dict[key] = torch.from_numpy(
                        kwargs[key]).float().cuda()
                else:
                    self._data_dict[key] = torch.from_numpy(kwargs[key])

        if 'action' in self._data_dict.keys():
            if self._data_dict['action'].dim() == 1:
                for key in self._data_dict:
                    self._data_dict[key] = torch.unsqueeze(
                        torch.squeeze(self._data_dict[key]), 0)

    def __len__(self):
        return len(self._data_dict[list(self._data_dict.keys())[0]])

    def __str__(self):
        string = ""
        for key in self._data_dict:
            string += key
            string += ":\n " + str(self._data_dict[key]) + "\n"
        return string

    @property
    def state(self) -> Tensor:
        """Get state

        :return: state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._data_dict['state']

    @property
    def action(self) -> Tensor:
        """Get action

        :return: action
        :rtype: Tensor[data_size, action_dim]
        """
        return self._data_dict['action']

    @property
    def next_state(self) -> Tensor:
        """Get the next state

        :return: next state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._data_dict['next_state']

    @property
    def reward(self) -> Tensor:
        """Get reward

        :return: reward
        :rtype: Tensor[data_size]
        """
        return self._data_dict['reward']

    @property
    def done_flag(self) -> Tensor:
        """Get done flag

        :return: Done flag 
        :rtype: Tensor[data_size]
        """
        return self._data_dict['done_flag']

    def cat(self, datas: Tuple[Data, ...]) -> Data:
        """Cat the current Data instance with the other Data instances, and return a new Data instance.

        :return: The new Data instance
        :rtype: Data
        """
        new_data = self.clone()
        if isinstance(datas, tuple):
            for data in datas:
                if data._data_dict.keys() == self._data_dict.keys():
                    for key in data._data_dict:
                        new_data._data_dict[key] = torch.torch.cat(
                            [new_data._data_dict[key], data._data_dict[key]], dim=0)
                else:
                    raise Exception(
                        "Cannot perform \"cat\" among Data instances with different keys!")
        else:
            if datas._data_dict.keys() == self._data_dict.keys():
                for key in datas._data_dict:
                    new_data._data_dict[key] = torch.torch.cat(
                        [new_data._data_dict[key], datas._data_dict[key]], dim=0)
            else:
                raise Exception(
                    "Cannot perform \"cat\" among Data instances with different keys!")
        return new_data

    def clone(self) -> Data:
        """Clone a new Data instance with the same content on the same device.

        :return: The new Data instance
        :rtype: Data
        """
        return copy.deepcopy(self)
