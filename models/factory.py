from abc import ABC, abstractmethod


class NetworkFactory(ABC):

    @abstractmethod
    def define(self, *args):
        pass


class OptimizerFactory(ABC):

    @abstractmethod
    def define(self, *args):
        pass


class LossFactory(ABC):

    @abstractmethod
    def define(self, *args):
        pass


class ArchitectFactory(ABC):

    @abstractmethod
    def define_network(self, *args):
        pass

    @abstractmethod
    def define_optimizer(self, *args):
        pass

    @abstractmethod
    def define_loss(self, *args):
        pass


