

from store_vectors import get_extract_model, extract_vector
import pickle
from matplotlib import pyplot as plt
import faiss
import sys

# Thiết lập encoding utf-8 cho stdout
sys.stdout.reconfigure(encoding='utf-8')


loaded_index = faiss.read_index("training-faiss.index")

path_images = pickle.load(open('paths.pkl', 'rb'))

image_path_search = 'dataset/2.jpg'


model = get_extract_model()

vector_search = extract_vector(model, image_path_search)
vector_search = vector_search.reshape(1, -1)


k = 12

D, I = loaded_index.search(vector_search, k)

print("Search result:")
print("Distance", D)
print("k-NN", I)

for neighbor_idx, distance in zip(I[0], D[0]):
    # Nearest neighbours' locations
    neighbor_image_path = path_images[neighbor_idx]
    neighbor_image = plt.imread(neighbor_image_path)  # Read images

    # Displaying neighbor
    plt.figure()
    plt.imshow(neighbor_image)
    plt.title("Nearest neighbor - Distance: {}".format(distance))
    plt.show()
