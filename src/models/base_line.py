import pandas as pd
from evaluate import *
from sklearn.metrics import mean_absolute_error
from preprocess import encode_user_item, random_split

import warnings
warnings.filterwarnings('ignore')

def base_line(df_ratings = None, query = 0, name = 'movie'):
    if df_ratings is None: df_ratings = pd.read_csv('../resource/ratings.csv')

    data, _, _ = encode_user_item(df_ratings)
    
    train, test = random_split(data, [0.75, 0.25])
    
    predictions_ratings = average_rating_model(train)
    predictions_ranking = popular_item_model(train)
    
    rating_evaluate_df = pd.merge(test, predictions_ratings, on=['user_index'], how='inner')
    ranking_top_k = recommend_top_k(data, train, predictions_ranking, 10)

    eval_precision = precision_at_k(test, ranking_top_k, k = 10)
    eval_recall = recall_at_k(test, ranking_top_k, k = 10)
    eval_ndcg = ndcg_at_k(test, ranking_top_k, k = 10)

    return ranking_top_k[:5], eval_precision, eval_recall, eval_ndcg, 0

def average_rating_model(train):
    users_ratings = train.groupby(["user_index"])["rating"].mean()
    users_ratings = users_ratings.reset_index()
    users_ratings.rename(columns = {'rating': 'rating_pred'}, inplace = True)

    return users_ratings

def popular_item_model(train):
    item_counts = (train.groupby("item_index")
                   .count()
                   .reset_index()
                   .sort_values(ascending = False, by = "user_index"))
    item_counts = item_counts[["item_index", "user_index"]]
    item_counts.columns = ['item_index', 'rating_pred']
        
    return item_counts

def recommend_top_k(data, train, predictions_ranking,k=5):
    user_item = user_item_crossjoin(data)
    all_predictions = pd.merge(user_item, predictions_ranking, on="item_index", how="left")
    all_predictions.fillna(0, inplace=True)
    all_predictions_unseen = filter_by(all_predictions, train, ["user_index", "item_index"])
    recommend_topk_df = get_top_k_items(all_predictions_unseen, "user_index", "rating_pred", k=5)
    return recommend_topk_df
