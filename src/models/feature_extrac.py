import pickle
import pandas as pd
from recommend import get_similar
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings('ignore')

def show_similar(item_index, item_similar_indice):
   movie_ids = item_similar_indice[item_index]
   images = []
   for movie_id in movie_ids:
      img_path = '../resource/posters/' + str(movie_id) + '.jpg'
      images.append(mpimg.imread(img_path))
   fig = plt.figure(figsize = (20,10))
   columns = 5
   for i, image in enumerate(images):
      fig.add_subplot(len(images) / columns + 1, columns, i + 1)
      plt.axis('off')
      plt.imshow(image)
   plt.show()

def feature_extraction(query = 0, name = 'movie'):
   items_raw = pd.read_csv('../resource/items_raw.csv')
   items_feature = pd.read_csv('../resource/item_features.csv')

   try:
      items = pd.read_csv(f'../cache/{name}/feature_ext/item.csv')
   except FileNotFoundError:
      items_raw['release_date'] = pd.to_datetime(items_raw['release_date'], infer_datetime_format = True)
      items_raw['year'] = items_raw['release_date'].apply(lambda x: str(x.year))
      
      items_main = items_raw.drop(['video_release_date', 'release_date', 'imdb_url'], axis=1).copy()
      items_addtl = items_feature[['overview', 'original_language', 'runtime', 'vote_average', 'vote_count', "movie_id"]].copy()
      items = pd.merge(left=items_main,right=items_addtl, on = 'movie_id', how='left')
      
      items['overview'].fillna('None', inplace=True)
      items.to_csv(f'../cache/{name}/feature_ext/item.csv', index=False)

   try:
      overview_embedding_df = pd.read_csv(f'../cache/{name}/feature_ext/overview_embedding_df.csv')
   except FileNotFoundError:
      sentence_tokens = [word_tokenize(text.lower()) for text in items['overview']]
      model = Word2Vec(sentences = sentence_tokens, min_count = 1)

      overview_embedding_list = []
      try:
         for vec in model.wv:
            overview_embedding_list.append(vec)
      except KeyError:
         pass

      overview_embedding_df = pd.DataFrame(overview_embedding_list)
      overview_embedding_df.to_csv(f'../cache/{name}/feature_ext/overview_embedding_df.csv', index=False)
   
   item_similarity_df = pd.concat([
      items[['movie_id', 'genre_unknown', 'Action', 'Adventure',
             'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
             'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance',
             'SciFi', 'Thriller', 'War', 'Western']], overview_embedding_df], axis=1)

   try:
      item_distance = pickle.load(open(f'../cache/{name}/feature_ext/item_distance.pkl', 'rb'))
      item_similar_indices = pickle.load(open(f'../cache/{name}/feature_ext/item_similar_indices.pkl', 'rb'))
   except FileNotFoundError:
      item_distance, item_similar_indices = get_similar(overview_embedding_df, 5)
      pickle.dump(item_distance, open(f'../cache/{name}/feature_ext/item_distance.pkl', 'wb'))
      pickle.dump(item_similar_indices, open(f'../cache/{name}/feature_ext/item_similar_indices.pkl', 'wb'))
   
   show_similar(query, item_similar_indices)