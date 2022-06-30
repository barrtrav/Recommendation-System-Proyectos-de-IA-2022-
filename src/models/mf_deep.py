import pandas as pd
import pickle
from preprocess import encode_user_item, user_split, random_split
from evaluate import *
from recommend import get_similar, show_similar
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.layers import Input, Embedding, Flatten, Dot, Add, Lambda, Activation, Concatenate, Dense, Dropout

import warnings
warnings.filterwarnings('ignore')

def deep_mf(n_users, n_items, n_factors, max_rating, min_rating):
    item_input = Input(shape=[1], name='item_index')
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6), embeddings_initializer = 'glorot_normal', name='item_embedding')(item_input)
    item_vec = Flatten(name='flatten_item_e')(item_embedding)

    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-6), embeddings_initializer = 'glorot_normal', name='item_bias')(item_input)
    item_bias_vec = Flatten(name='flatten_item_bias_e')(item_bias)

    user_input = Input(shape=[1], name='user_index')
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6), embeddings_initializer = 'glorot_normal', name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_user_e')(user_embedding)

    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5), embeddings_initializer = 'glorot_normal', name='user_bias')(user_input)
    user_bias_vec = Flatten(name='flatten_user_bias_e')(user_bias)

    concat = Concatenate(name='concat')([item_vec, user_vec])
    concat_drop = Dropout(0.5)(concat)

    kernel_initializer = 'he_normal'

    dense_1 = Dense(10, kernel_initializer='glorot_normal', name='dense_1')(concat_drop)
    dense_1_drop = Dropout(0.5)(dense_1)
    dense_2 = Dense(1, kernel_initializer='glorot_normal', name='dense_2')(dense_1_drop)

    add_bias = Add(name="add_bias")([dense_2, item_bias_vec, user_bias_vec])

    y = Activation('sigmoid')(add_bias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)

    model = Model([user_input, item_input], rating_output)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    
    return model

def matrix_factorisation_deep(df_ratings = None, query = 0, name = 'movie'):
    if df_ratings is None: df_ratings = pd.read_csv('../resource/ratings.csv')

    data, user_encoder, item_encoder = encode_user_item(df_ratings)

    n_users = data['user_index'].nunique()
    n_items = data['item_index'].nunique()
    
    max_rating = data['rating'].max()
    min_rating = data['rating'].min()

    train, test = random_split(data, [0.9, 0.1])

    try:
        n_factors = 50
        model = load_model(f'../cache/{name}/mf_deep/model.tf')
    except OSError:
        model = deep_mf(n_users, n_items, n_factors, max_rating, min_rating) 
        output = model.fit([train['user_index'], train['item_index']], train['rating'],
                            batch_size=128, epochs=5, verbose=1, validation_data=([test['user_index'], test['item_index']], test['rating']))
        model.save(f'../cache/{name}/mf_deep/model.tf')

    score = model.evaluate([test['user_index'], test['item_index']], test['rating'], verbose=1)

    item_embedding = get_embedding(model, 'item_embedding')
    user_embedding = get_embedding(model, 'user_embedding')

    try:
        predictions = pd.read_csv(f'../cache/{name}/mf_deep/predict.csv')
    except FileNotFoundError:
        predictions = get_predictions(model, data)
        predictions.to_csv(f'../cache/{name}/mf_deep/predict.csv')

    try:
        ranking_top = pd.read_csv(f'../cache/{name}/mf_deep/ranking.csv')
    except FileNotFoundError:
        ranking_top = recommend_topk(model, data, train, k=5)
        ranking_top.to_csv(f'../cache/{name}/mf_deep/ranking.csv')
    
    eval_precision = precision_at_k(test, ranking_top, k=10)
    eval_recall = recall_at_k(test, ranking_top, k=10)
    eval_ndcg = ndcg_at_k(test, ranking_top, k=10)

    try:
        item_distance = pickle.load(open(f'../cache/{name}/mf_deep/item_distance.pkl', 'rb'))
        item_similar_indices = pickle.load(open(f'../cache/{name}/mf_deep/item_similar_indices.pkl', 'rb'))
    except FileNotFoundError:
        item_distance, item_similar_indices = get_similar(item_embedding, 5)
        pickle.dump(item_distance, open(f'../cache/{name}/mf_deep/item_distance.pkl', 'wb'))
        pickle.dump(item_similar_indices, open(f'../cache/{name}/mf_deep/item_similar_indices.pkl', 'wb'))

    if name == 'movie': show_similar(query, item_similar_indices, item_encoder)

    return ranking_top[:5], eval_precision, eval_recall, eval_ndcg, score