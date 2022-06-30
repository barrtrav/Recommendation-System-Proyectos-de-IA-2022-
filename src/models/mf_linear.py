import view
import pickle
import pandas as pd
from evaluate import *
from recommend import get_similar, show_similar
from preprocess import encode_user_item, random_split

from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot

import warnings
warnings.filterwarnings('ignore')

def explicit_mf(n_users, n_items, n_factors):
    item_input = Input(shape=[1], name='item')
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                                name='item_embedding')(item_input)
    item_vec = Flatten(name='flatten_items_e')(item_embedding)
        
    user_input = Input(shape=[1], name='user')
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6),
                                name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_user_e')(user_embedding)

    rating = Dot(axes=1, name='dot_product')([item_vec, user_vec])

    model = Model([user_input, item_input], rating)
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model

def matrix_factorisation_linear(df_ratings = None, query = 0, name = 'movie'):
    if df_ratings is None: df_ratings = pd.read_csv('../resource/ratings.csv')

    data, user_encoder, item_encoder = encode_user_item(df_ratings)

    n_users = data['user_index'].nunique()
    n_items = data['item_index'].nunique()
    
    max_rating = data['rating'].max()
    min_rating = data['rating'].min()

    train, test = random_split(data, [0.8, 0.2])

    try:
        n_factors = 40
        model = load_model(f'../cache/{name}/mf_linear/model.tf')
    except OSError:
        model = explicit_mf(n_users, n_items, n_factors) 
        output = model.fit([train['user_index'], train['item_index']], train['rating'],
                            batch_size=128, epochs=5, verbose=1, validation_split=0.2)
        model.save(f'../cache/{name}/mf_linear/model.tf')

    score = model.evaluate([test['user_index'], test['item_index']], test['rating'], verbose=1)

    item_embedding = get_embedding(model, 'item_embedding')
    user_embedding = get_embedding(model, 'user_embedding')

    try:
        predictions = pd.read_csv(f'../cache/{name}/mf_linear/predict.csv')
    except FileNotFoundError:
        predictions = get_predictions(model, data)
        predictions.to_csv(f'../cache/{name}/mf_linear/predict.csv')

    try:
        ranking_top = pd.read_csv(f'../cache/{name}/mf_linear/ranking.csv')
    except FileNotFoundError:
        ranking_top = recommend_topk(model, data, train, k=5)
        ranking_top.to_csv(f'../cache/{name}/mf_linear/ranking.csv')
    
    eval_precision = precision_at_k(test, ranking_top, k=10)
    eval_recall = recall_at_k(test, ranking_top, k=10)
    eval_ndcg = ndcg_at_k(test, ranking_top, k=10)

    try:
        item_distance = pickle.load(open(f'../cache/{name}/mf_linear/item_distance.pkl', 'rb'))
        item_similar_indices = pickle.load(open(f'../cache/{name}/mf_linear/item_similar_indices.pkl', 'rb'))
    except FileNotFoundError:
        item_distance, item_similar_indices = get_similar(item_embedding, 5)
        pickle.dump(item_distance, open(f'../cache/{name}/mf_linear/item_distance.pkl', 'wb'))
        pickle.dump(item_similar_indices, open(f'../cache/{name}/mf_linear/item_similar_indices.pkl', 'wb'))

    if name == 'movie': show_similar(query, item_similar_indices, item_encoder)

    return ranking_top[:5], eval_precision, eval_recall, eval_ndcg, score