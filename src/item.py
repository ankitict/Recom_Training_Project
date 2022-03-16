import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader
from surprise import KNNWithMeans
from surprise import dump
import json

class Item_Recommendation:
    """
        This class train KNNWithMean Model for Item Filtering and finding similar items

        Written By: Ankit Pansuriya
        Version: 1.0
        Revisions: None
    """

    def filter_item_ids(data, algo, trainset):

        item_data = data
        product_categories = list(item_data["product_category"].value_counts().index)

        item_id_mapping = {}

        for category in product_categories:
            _id = algo.trainset.to_inner_iid(category)
            item_id_mapping[category] = _id

        file_path = Path.cwd() / "./datasets/item_id_mapping.json"
        json.dump(item_id_mapping, open(file_path, 'w'))
        return

    def train_item_reccommendation_model():
        """
            Method Name: train_item_reccommendation_model
            Description: This method train KNN with Mean model
            Output: Dump KNN model to the .pkl file
            On Failure: Raise Exception

            Written By: Ankit Pansuriya
            Version: 1.0
            Revisions: None
        """
        item_data = pd.read_csv("./datasets/item_data.csv")

        # Creating Data object.
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df=item_data, reader=reader)
        trainset = data.build_full_trainset()

        # Training Algorithm.
        sim_options = {
            "name": "cosine",
            "user_based": False
        }
        algo = KNNWithMeans(k=10, sim_options=sim_options, verbose=False)
        algo.fit(trainset)

        # Extract inner id mappings.
        Item_Recommendation.filter_item_ids(item_data, algo=algo, trainset=trainset)

        # Saving Algorithm.
        file_path = Path.cwd() / "model/similar_items.pkl"
        dump.dump(str(file_path), algo=algo)
        return