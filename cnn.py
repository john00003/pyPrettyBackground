import torch
from torchvision import models, transforms

class resNet50Model():
    def __init__(self):
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final layer - gives us feature vector instead of classification
        self.model = model
        #model.eval()

    def extract_features(self, x):


        with torch.no_grad():
            feature_vector = self.model(x)
        return feature_vector.view(x.size(0), -1).numpy() # return flattened feature vector as numpy array




