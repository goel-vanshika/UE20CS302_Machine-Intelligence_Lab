from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np


class SVM:
    def __init__(self, dp):
        self.dp = dp
        data = pd.read_csv(self.dp)
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]
    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """
        # TODO
        s = SVC(C=1.0, random_state=1, kernel='rbf')
        steps = [('scaler',StandardScaler()),('svc',s)]
        pl = Pipeline(steps)
        pl.fit(self.X, self.y)
        return pl