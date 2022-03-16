import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.dump import dump
from pathlib import Path


class User_Recommendation:
    def train_user_reccommendation_model():
        """
            Method Name: train_user_reccommendation_model
            Description: This method train KNN with Mean model
            Output: Dump KNN model to the .pkl file
            On Failure: Raise Exception

            Written By: Ankit Pansuriya
            Version: 1.0
            Revisions: None
        """
        user_rec_sys_data = pd.read_csv("./datasets/user_data.csv")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(user_rec_sys_data, reader)
        trainset = data.build_full_trainset()

        # Training.
        algo = SVD()
        algo.fit(trainset)

        # Saving Algorithm.
        file_path = Path.cwd() / "model/user_pred.pkl"
        dump(str(file_path), algo=algo)
        return

