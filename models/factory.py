from abc import ABC, abstractmethod


class NetworkFactory(ABC):

    @abstractmethod
    def define_model(param, *args):
        pass

    @abstractmethod
    def define_optimizer(param, *args):
        pass

    @abstractmethod
    def define_loss(loss_type):
        pass
