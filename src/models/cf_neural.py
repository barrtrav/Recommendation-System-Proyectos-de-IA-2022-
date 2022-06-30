import pandas as pd
import numpy as np
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

def neural_cf(n_users, n_items, n_factors, max_rating, min_rating):
    item_input = Input(shape=[1], name='item')
    item_embedding_mf = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                                  embeddings_initializer='he_normal',
                                  name='item_embedding_mf')(item_input)
    item_vec_mf = Flatten(name='flatten_item_mf')(item_embedding_mf)
    item_embedding_mlp = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                                embeddings_initializer='he_normal',
                               name='item_embedding_mlp')(item_input)
    item_vec_mlp = Flatten(name='flatten_item_mlp')(item_embedding_mlp)

    user_input = Input(shape=[1], name='user_index')
    user_embedding_mf = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6), 
                                embeddings_initializer='he_normal',
                               name='user_embedding_mf')(user_input)
    user_vec_mf = Flatten(name='flattenUserMF')(user_embedding_mf)
    user_embedding_mlp = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer='he_normal',
                               name='user_embedding_mlp')(user_input)
    user_vec_mlp = Flatten(name='flattenUser_mlp')(user_embedding_mlp)

    dot_product_mf = Dot(axes=1, name='dot_product_mf')([item_vec_mf, user_vec_mf])
    

    concat_mlp = Concatenate(name='concat_mlp')([item_vec_mlp, user_vec_mlp])

    dense_1 = Dense(50, name="dense_1")(concat_mlp)
    dense_2 = Dense(20, name="dense_2")(dense_1)

    concat = Concatenate(name="concat_all")([dot_product_mf, dense_2])
    
    pred = Dense(1, name="pred")(concat)

    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-5), name='item_bias')(item_input)
    item_bias_vec = Flatten(name='flatten_item_bias_e')(item_bias)

    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5), name='user_bias')(user_input)
    user_bias_vec = Flatten(name='flatten_user_bias_e')(user_bias)

    pred_add_bias = Add(name="add_bias")([pred, item_bias_vec, user_bias_vec])

    y = Activation('sigmoid')(pred_add_bias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)

    model = Model([user_input, item_input], rating_output)
    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model

def deep_neural_cf(df_ratings = None, query = 0, name = 'movie'):
    if df_ratings is None: df_ratings = pd.read_csv('../resource/ratings.csv')

    data, user_encoder, item_encoder = encode_user_item(df_ratings)

    n_users = data['user_index'].nunique()
    n_items = data['item_index'].nunique()
    
    max_rating = data['rating'].max()
    min_rating = data['rating'].min()

    train, test = user_split(data, [0.8, 0.2])

    try:
        n_factors = 40
        model = load_model(f'../cache/{name}/cf_neural/model.tf')
    except OSError:
        model = neural_cf(n_users, n_items, n_factors, max_rating, min_rating) 
        output = model.fit([train['user_index'], train['item_index']], train['rating'],
                            batch_size=128, epochs=5, verbose=1, validation_split=0.2)
        model.save(f'../cache/{name}/cf_neural/model.tf')

    score = model.evaluate([test['user_index'], test['item_index']], test['rating'], verbose=1)

    item_embedding_mf = get_embedding(model, 'item_embedding_mf')
    user_embedding_mf = get_embedding(model, 'user_embedding_mf')
    item_embedding_mlp = get_embedding(model, 'item_embedding_mlp')
    user_embedding_mlp = get_embedding(model, 'user_embedding_mlp')

    item_embedding = np.mean([item_embedding_mf, item_embedding_mlp], axis=0)
    user_embedding = np.mean([user_embedding_mf, user_embedding_mlp], axis=0)

    try:
        predictions = pd.read_csv(f'../cache/{name}/cf_neural/predict.csv')
    except FileNotFoundError:
        predictions = get_predictions(model, data)
        predictions.to_csv(f'../cache/{name}/cf_neural/predict.csv')

    try:
        ranking_top = pd.read_csv(f'../cache/{name}/cf_neural/ranking.csv')
    except FileNotFoundError:
        ranking_top = recommend_topk(model, data, train, k=5)
        ranking_top.to_csv(f'../cache/{name}/cf_neural/ranking.csv')
    
    eval_precision = precision_at_k(test, ranking_top, k=10)
    eval_recall = recall_at_k(test, ranking_top, k=10)
    eval_ndcg = ndcg_at_k(test, ranking_top, k=10)

    try:
        item_distance = pickle.load(open(f'../cache/{name}/cf_neural/item_distance.pkl', 'rb'))
        item_similar_indices = pickle.load(open(f'../cache/{name}/cf_neural/item_similar_indices.pkl', 'rb'))
    except FileNotFoundError:
        item_distance, item_similar_indices = get_similar(item_embedding, 5)
        pickle.dump(item_distance, open(f'../cache/{name}/cf_neural/item_distance.pkl', 'wb'))
        pickle.dump(item_similar_indices, open(f'../cache/{name}/cf_neural/item_similar_indices.pkl', 'wb'))

    if name == 'movie': show_similar(query, item_similar_indices, item_encoder)

    return ranking_top[:5], eval_precision, eval_recall, eval_ndcg, score