from abc import ABC, abstractmethod
from onnx import ModelProto

class ModelInteface(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> dict:
        pass

