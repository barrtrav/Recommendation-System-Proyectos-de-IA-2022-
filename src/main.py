import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from models.base_line import base_line
from models.cf_neural import deep_neural_cf
from models.mf_deep import matrix_factorisation_deep
from models.feature_extrac import feature_extraction
from models.mf_linear import matrix_factorisation_linear
from models.mf_non_neg import matrix_factorisation_non_negative
from models.mf_linear_bias import matrix_factorisation_linear_bias

def print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score):
    print('Ranking top:')
    print(ranking_top[['rank', 'item_index',  'rating_pred']])
    print()
    print('Precision:', eval_precision)
    print('Recall:', eval_recall)
    print('NDCG:', eval_ndcg)
    print()
    print('Score:', score)


print('''
Choose a model:
1: Model - Baseline
2: Model - Matrix Factorisation - Linear
3: Model - Matrix Factorisation - Linear with Bias
4: Model - Matrix Factorisation - Non Negative MF
5: Model - Deep Matrix Factorisation
6: Model - Deep Neural Collaborative Filtering
----------------------------------------------------
7: Feature extractor

''')

while True:
    model = int(input('Enter option number:'))
    if model >= 1 and model <= 7:break

print('''
Choose a data set:
1: Model - Movies
2: Model - Products

''')

while True:
    ds = int(input('Enter option number:'))
    if ds == 1 or model == 2:break

print()
user = int(input('Enter number of item:'))

if ds == 2 and model != 7 :
    name = 'product'
    df_ratings = pd.read_csv('../resource/ratings_Beauty.csv')
else:
    name = 'movie'
    df_ratings = None

if model == 1:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = base_line(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 2:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = matrix_factorisation_linear(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 3:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = matrix_factorisation_linear_bias(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 4:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = matrix_factorisation_non_negative(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 5:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = matrix_factorisation_deep(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 6:
    ranking_top, eval_precision, eval_recall, eval_ndcg, score = deep_neural_cf(df_ratings, user, name)
    print_result(ranking_top, eval_precision, eval_recall, eval_ndcg, score)
if model == 7:
    feature_extraction(user, name)