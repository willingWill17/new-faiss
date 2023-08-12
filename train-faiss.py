import numpy as np
from store_vectors import vectors, paths

import faiss
import pickle

vectors_2d = np.array(vectors)
index = faiss.IndexFlatL2(vectors_2d.shape[1])
index.add(vectors_2d)

faiss.write_index(index, "training-faiss.index")

pickle.dump(paths, open('paths.pkl', 'wb'))
