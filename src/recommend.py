import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors

def get_similar(embedding, k):
    model_similar_items = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embedding)
    distance, indece = model_similar_items.kneighbors(embedding)
    return distance, indece

def show_similar(item_index, item_similar_indices, item_encoder):
        
    s = item_similar_indices[item_index]
    movie_ids = item_encoder.inverse_transform(s)

    images = []
    for movie_id in movie_ids:
        img_path = '../resource/posters/' + str(movie_id) + '.jpg'
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.axis('off')
        plt.imshow(image)
    plt.show()