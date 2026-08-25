from . import BaseClass
import os
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris

class ONNXExport(BaseClass):
    def __init__(self):
        super().__init__()

    def convert_to_onnx(self):
        self.make_artifact_folder()
        pkl_path = os.path.join(self.artifact_folder, "model.pkl")
        onnx_path = os.path.join(self.artifact_folder, "model.onnx")

        X = load_iris().data

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]

        onnx_model = convert_sklearn(
            model, initial_types=initial_type, target_opset=17
        )

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        