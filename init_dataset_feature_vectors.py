import pandas as pd
from os import listdir
from os.path import join
from cnn import resNet50Model
from process_image import load_image
import numpy as np

class DatasetFeatureVectorManager:
    def __init__(self, dataset_path="C:\\Users\\doggo\\Pictures\\lhq_dataset", feature_vectors_path="C:\\github repos\\pyPrettyBackground"):
        self.dataset_path = dataset_path
        self.feature_vectors_path = feature_vectors_path
        self.feature_vectors_df = None
        self.model = None

    def get_dataset_feature_vector_df(self):
        if self.feature_vectors_df is None:
            self.feature_vectors_df = self.init_dataset_feature_vectors()
        return self.feature_vectors_df

    def init_dataset_feature_vectors(self):
        # check if we have already initialized the .csv file
        if ('dataset_feature_vectors.csv' in listdir(self.feature_vectors_path)):
            print("Reading CSV file...")
            df = pd.read_csv('dataset_feature_vectors.csv')
            return df
        else:
            # get all file names, then use CNN to get feature vector for each image
            file_names = [join(self.dataset_path, f) for f in listdir(self.dataset_path)]

            dataset_len = len(file_names)
            print("Initializing feature vector for all images in the dataset. This may take a while.")
            print("There are {0} files in total in the dataset".format(dataset_len))

            all_feature_vectors = []
            percent_complete = 0
            previous_percent_complete = 0

            for i in range(dataset_len):
                all_feature_vectors.append(self.get_image_feature_vector(file_names[i]))
                percent_complete += 1/dataset_len
                if percent_complete >= previous_percent_complete + 0.01:
                    print("{0}%".format(percent_complete*100))
                    previous_percent_complete = percent_complete


            #all_feature_vectors = [self.get_image_feature_vector(image) for image in file_names]
            print("Squeezing feature vector array")
            all_feature_vectors = np.squeeze(np.array(all_feature_vectors))

            print("Creating the pandas dataframe.")
            print("Converting NumPy array to a list.")
            print(np.shape(all_feature_vectors))
            d = {"feature_vectors": all_feature_vectors.tolist()}
            print("Inserting feature vectors.")
            df = pd.DataFrame(d)
            print("Inserting file names.")
            df.insert(0, 'file_name', file_names)
            print("Saving the dataframe as csv.")
            df.to_csv('dataset_feature_vectors.csv', index=False)

        return df

    def get_image_feature_vector(self, file_name):
        if self.model is None:
            self.initialize_model()

        return self.model.extract_features((load_image(file_name)))

    def initialize_model(self):
        self.model = resNet50Model()