import os
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .BaseClass import BaseClass


class TrainClass(BaseClass):
    def __init__(self):
        super().__init__()

    
    def train_and_save_model(self):
        # 1. Load Dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # 2. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # 3. Train Model
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        # 4. Save Artifact
        pkl_path = os.path.join(self.artifact_folder, "model.pkl") 

        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

