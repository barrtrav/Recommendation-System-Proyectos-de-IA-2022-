import numpy as np
import pandas as pd

def get_embedding(model, name):
    embedding = model.get_layer(name = name).get_weights()[0]
    return embedding

def get_predictions(model, data):
    user_item = user_item_crossjoin(data)
    user_item["rating_pred"] = model.predict([user_item['user_index'], user_item['item_index']])
    return user_item

def user_item_crossjoin(df):
    crossjoin_list = []
    for user in df['user_index'].unique():
        for item in df['item_index'].unique():
            crossjoin_list.append([user, item])
    cross_join_df = pd.DataFrame(data=crossjoin_list, columns=["user_index", "item_index"])
    return cross_join_df
    
def filter_by(df, filter_by_df, filter_by_cols):
    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(
            filter_by_df.set_index(filter_by_cols).index
        )
    ]

def get_top_k_items(df, col_user, col_rating, k=10):
    top_k_items = (
        df.groupby(col_user, as_index=False)
        .apply(lambda x: x.nlargest(k, col_rating))
        .reset_index(drop=True)
    )
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items

def recommend_topk(model, data, train, k=5):
    all_predictions = get_predictions(model, data)
    all_predictions.fillna(0, inplace=True)
    all_predictions_unseen = filter_by(all_predictions, train, ["user_index", "item_index"])
    recommend_topk_df = get_top_k_items(all_predictions_unseen, "user_index", "rating_pred", k=5)
    return recommend_topk_df

def get_hit_df(rating_true, rating_pred, k):
    common_users = set(rating_true["user_index"]).intersection(set(rating_pred["user_index"]))
    rating_true_common = rating_true[rating_true["user_index"].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred["user_index"].isin(common_users)]
    n_users = len(common_users)
    df_hit = get_top_k_items(rating_pred_common, "user_index", "rating_pred", k)
    df_hit = pd.merge(df_hit, rating_true_common, on=["user_index", "item_index"])[
        ["user_index", "item_index", "rank"]
    ]
    df_hit_count = pd.merge(
        df_hit.groupby("user_index", as_index=False)["user_index"].agg({"hit": "count"}),
        rating_true_common.groupby("user_index", as_index=False)["user_index"].agg(
            {"actual": "count"}
        ),
        on="user_index",
    )
    return df_hit, df_hit_count, n_users

def precision_at_k(rating_true, rating_pred, k):
    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    if df_hit.shape[0] == 0:
        return 0.0
    return (df_hit_count["hit"] / k).sum() / n_users

def recall_at_k(rating_true, rating_pred, k):
    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    if df_hit.shape[0] == 0:
        return 0.0
    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users

def ndcg_at_k(rating_true, rating_pred, k):
    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    if df_hit.shape[0] == 0:
        return 0.0
    df_dcg = df_hit.copy()
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    df_dcg = df_dcg.groupby("user_index", as_index=False, sort=False).agg({"dcg": "sum"})
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=["user_index"])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users