import numpy as np
from init_dataset_feature_vectors import DatasetFeatureVectorManager
from ast import literal_eval

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def find_most_similar(feature_vector, similarity_function=cosine_similarity):
    max = -1
    max_index = 0
    dataset_manager = DatasetFeatureVectorManager()
    dataset_feature_vectors_df = dataset_manager.get_dataset_feature_vector_df()
    index = 0
    print("Finding most similar features in dataset. This may take a while.")
    percent_complete = 0
    previous_percent_complete = 0
    dataset_len = len(dataset_feature_vectors_df["feature_vectors"])
    for dataset_feature_vector in dataset_feature_vectors_df["feature_vectors"]:
        #dataset_feature_vector = np.asarray(dataset_feature_vector, dtype=float)
        dataset_feature_vector = literal_eval(dataset_feature_vector)
        similarity = similarity_function(dataset_feature_vector, feature_vector)
        if similarity > max:
            max = similarity
            max_index = index
        index += 1
        percent_complete += 1 / dataset_len
        if percent_complete >= previous_percent_complete + 0.01:
            print("{0}%".format(percent_complete * 100))
            previous_percent_complete = percent_complete

    max_similarity_row = dataset_feature_vectors_df.iloc[max_index]
    return max_similarity_row