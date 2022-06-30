import numpy as np
from pandas import DataFrame, concat
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

def encode_user_item(df : DataFrame) -> DataFrame:#Tuple[DataFrame, LabelEncoder, LabelEncoder]:
    encode_df = df.copy()

    user_col = 'user_id'
    item_col = 'movie_id'

    user_encode = LabelEncoder()
    user_encode.fit(encode_df[user_col].values)

    item_encode = LabelEncoder()
    item_encode.fit(encode_df[item_col])

    encode_df['user_index'] = user_encode.transform(encode_df[user_col])
    encode_df['item_index'] = item_encode.transform(encode_df[item_col])

    encode_df.rename({'unix_timestamp' : 'timestamp'}, axis = 1, inplace = True)

    return encode_df, user_encode, item_encode

def random_split(df : DataFrame, ratios : List[float]) -> List[DataFrame]:
    samples = df.shape[0]

    split_ratio = np.cumsum(ratios).tolist()[:-1]
    split_index = [round(x * samples) for x in split_ratio]
    splits = np.split(df, split_index)
    
    for i in range(len(ratios)):
        splits[i]['split_index'] = i
    
    return splits

def user_split(df : DataFrame, ratios : List[float], chrono : bool = False) -> List[DataFrame]:
    splits = []

    user_col = 'user_id'
    time_col = 'timestep'
    
    if chrono:df_grouped = df.sort_values(time_col).groupby(user_col)
    else:df_grouped = df.groupby(user_col)

    for name, group in df_grouped:
        group_splits = random_split(df_grouped.get_group(name), ratios)
        concat_group_splits = concat(group_splits)
        splits.append(concat_group_splits)

    splits_all = concat(splits)
    splits_list = [ splits_all[splits_all['split_index'] == x] for x in range(len(ratios))]
    
    return splits_list # train val test