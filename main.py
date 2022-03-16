from src.data_preprocessing import Preprocessor
from src.Top_Trending_Items import Trending
from src.item import Item_Recommendation
from src.user import User_Recommendation

if __name__ == '__main__':
    # Preprocess the dataset
    Preprocessor.prepare_datasets()

    # Pick Top 10 trending items
    Trending.pick_top_ten_items()

    # Item Recommendation Using KNNWithMeans Model
    Item_Recommendation.train_item_reccommendation_model()

    # User Recommendation Using KNNWithMeans Model
    User_Recommendation.train_user_reccommendation_model()

