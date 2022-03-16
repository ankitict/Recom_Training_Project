import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import joblib
import random


class Preprocessor:
    """
            This class performs data preprocessing
                - Merge multiple datasets in to One.
                - returns cleaned data csv file for E-commerce Items
                - returns cleaned data csv file for E-commerce Users

            Written By: Ankit Pansuriya
            Version: 1.0
            Revisions: None
    """

    def prepare_datasets():
        """
            Method Name: prepare_datasets
            Description: This method prepare datasets from multiple csv files.
            Output: A pandas DataFrame after removing the specified columns.
            On Failure: Raise Exception

            Written By: Ankit Pansuriya
            Version: 1.0
            Revisions: None
        """
        try:
            # Read CSV files using Pandas
            customers_df = pd.read_csv("./datasets/olist_customers_dataset.csv")
            orders_df = pd.read_csv("./datasets/olist_orders_dataset.csv")
            reviews_df = pd.read_csv("./datasets/olist_order_reviews_dataset.csv")
            translation_df = pd.read_csv("./datasets/product_category_name_translation.csv")
            items_df = pd.read_csv("./datasets/olist_order_items_dataset.csv")
            products_df = pd.read_csv("./datasets/olist_products_dataset.csv")

            # Customers + Order Details
            data = customers_df[["customer_unique_id", "customer_id"]].copy()
            data = data.merge(orders_df[["order_id", "order_purchase_timestamp", "customer_id"]], on="customer_id",
                              how="left")

            data = data.merge(items_df[["order_id", "product_id"]], on="order_id", how="left")

            data = data.merge(reviews_df[["order_id", "review_score"]], on="order_id", how="left")

            products = products_df.merge(translation_df[["product_category_name", "product_category_name_english"]],
                                      on="product_category_name", how="left")

            data = data.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")

            data.drop_duplicates(inplace=True)

            review_data = data[["customer_unique_id", "product_category_name_english", "review_score"]].copy()
            review_data["product_category_name_english"].fillna("Others", axis=0,
                                                                inplace=True)

            review_data.rename(columns={"customer_unique_id": "customer_id",
                                        "product_category_name_english": "product_category"},
                               inplace=True)
            print("Data columns : ", data.columns)
            print("Review columns : ", review_data.columns)

            item_data = Preprocessor.Prepare_item_data(review_data)
            user_data = Preprocessor.Prepare_user_data(item_data)
            print(item_data.head())
            return
        except Exception as e:
            print("Exception Occurred while Prepare Datasets" + str(e))
            raise Exception()

    def Prepare_item_data(review_data):
        try:
            item_data = review_data.copy()
            # Encoding Customer Ids.
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(item_data[["customer_id"]])
            item_data["customer_id"] = encoded.astype("int").copy()

            file_path = Path.cwd() / "./model/cust_id_encoder.pkl"
            joblib.dump(encoder, file_path)

            file_path = Path.cwd() / "./datasets/item_data.csv"
            item_data.to_csv(file_path, index=False)

            return item_data
        except Exception as e:
            print("Exception Occurred while Prepare { Item } Dataset" + str(e))
            raise Exception()

    def Prepare_user_data(item_data):
        try:
            item_data = item_data.copy()
            review_counts = item_data.groupby("customer_id").count()[["review_score"]]
            all_ids = list(review_counts[review_counts["review_score"] > 1].index)

            random_sample_ids = random.choices(all_ids, k=100)
            user_data = item_data[item_data["customer_id"].isin(random_sample_ids)]

            user_data.reset_index(inplace=True)
            user_data = user_data.drop(["customer_id", "index"], axis=1)
            user_data.reset_index(inplace=True)
            user_data = user_data.rename(columns={"index": "customer_id"})

            file_path = Path.cwd() / "./datasets/user_data.csv"
            user_data.to_csv(file_path, index=False)
            return user_data
        except Exception as e:
            print("Exception Occurred while Prepare { User } Dataset" + str(e))
            raise Exception()