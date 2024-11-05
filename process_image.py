import numpy as np
from PIL import Image
from torchvision import transforms

def load_image(path):
    image = Image.open(path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def preprocess(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # # convert PIL image to an np array
    # image_arr = np.array(image)
    # # Convert RGB to BGR
    # image_arr = image_arr[:, :, ::-1].copy()
    #
    # # resize image into 224x224 using Lanczos Interpolation
    # image_arr = cv2.resize(image_arr, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)

    image = preprocess(image)
    return image

