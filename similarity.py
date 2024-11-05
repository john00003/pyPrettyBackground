import numpy as np
from init_dataset_feature_vectors import DatasetFeatureVectorManager



def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def find_most_similar(feature_vector, similarity_function=cosine_similarity):
    max = -1
    max_index = 0
    dataset_manager = DatasetFeatureVectorManager()
    dataset_feature_vectors_df = dataset_manager.get_dataset_feature_vector_df()
    index = 0
    for dataset_feature_vector in dataset_feature_vectors_df["feature_vectors"]:
        similarity = similarity_function(dataset_feature_vector, feature_vector)
        if similarity > max:
            max = similarity
            max_index = index
        index += 1

    max_similarity_row = dataset_feature_vectors_df.iloc[max_index]
    return max_similarity_row