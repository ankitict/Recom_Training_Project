import pandas as pd
from pathlib import Path

"""
Computes Weighted Average for each product.
Formula:
    WR = (v/v+m)*R+(m/v+m)*C
    v = Total Reviews
    m = Minimum votes required to qualify to trending list
    R = Average Rating
    C = Mean rating across the report.
"""

class Trending:
    """
        This class will return top 10 trending Items and dump to the json file

        Written By: Ankit Pansuriya
        Version: 1.0
        Revisions: None
    """

    def pick_top_ten_items():
        """
            Method Name: pick_top_ten_items
            Description: This method calculate weighted average of each product and Pick top ten out of it.
            Output: top ten trending product dump as a json file.
            On Failure: Raise Exception

            Written By: Ankit Pansuriya
            Version: 1.0
            Revisions: None
        """
        try:
            review_data = pd.read_csv("./datasets/item_data.csv")

            trending = review_data.groupby("product_category")[["review_score"]].agg(["mean", "count"])
            trending.columns = ["avg_review_score", "total_reviews"]

            c = trending["avg_review_score"].mean()

            m = trending["total_reviews"].quantile(0.9)

            # Weighted Average.
            trending["weighted_average"] = (trending["total_reviews"] / (trending["total_reviews"] + m)) * \
                                           (trending["avg_review_score"] + (m / (trending["total_reviews"] + m)) * c)

            # Top 10 Trending Items
            top_ten = trending.sort_values("weighted_average", ascending=False).head(10)
            top_ten = top_ten.drop("avg_review_score", axis=1)
            top_ten.reset_index(inplace=True)

            file_path = Path.cwd() / "./datasets/top_trending_items.json"
            top_ten.to_json(file_path)
            return
        except Exception as e:
            print("Exception Occurred while Prepare Datasets" + str(e))
            raise Exception()