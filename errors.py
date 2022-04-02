"""All custom exceptions raised by Trainer"""
import abc


class TrainerError(Exception):
    """Raised when Trainer encounters an error"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError


class InputError(TrainerError):
    """Raised when input is invalid"""

    def __init__(self, dim):
        self.msg = f"Dimensions of input are not identical but recieved {dim}"
