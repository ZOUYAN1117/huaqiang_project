from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    def infer(self, input_data):
        pass