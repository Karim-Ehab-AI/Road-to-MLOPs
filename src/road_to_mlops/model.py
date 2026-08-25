from abc import ABC, abstractmethod
from onnx import ModelProto
from .interfaces import ModelInteface

class IrisModel(ModelInteface):
    def __init__(self, model: ModelProto):
        self._model = model

    def predict(self, X: list[float]) -> dict:
        pass