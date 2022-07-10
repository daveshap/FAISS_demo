# pip install faiss-gpu

import numpy as np 
import faiss

dimension = 128    # dimensions of each vector                         
n = 200    # number of vectors                   
np.random.seed(1)             
db_vectors = np.random.random((n, dimension)).astype('float32')



nlist = 5  # number of clusters
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)



print(index.is_trained)   # False
index.train(db_vectors)  # train on the database vectors
print(index.ntotal)   # 0
index.add(db_vectors)   # add the vectors and update the index
print(index.is_trained)  # True
print(index.ntotal)   # 200


nprobe = 2  # find 2 most similar clusters
n_query = 10  
k = 3  # return 3 nearest neighbours
np.random.seed(0)   
query_vectors = np.random.random((n_query, dimension)).astype('float32')
distances, indices = index.search(query_vectors, k)


faiss.write_index(index,"vector.index")  # save the index to disk
index = faiss.read_index("vector.index")  # load the index

print(distances)

print(indices)