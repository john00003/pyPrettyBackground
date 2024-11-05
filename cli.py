from process_image import load_image
from cnn import resNet50Model
from similarity import find_most_similar
def prompt():
    selection_set = {1}
    while True:
        print("Please select an option:")
        print("1: Find a similar image")
        try:
            choice = int(input())
            if choice in selection_set:
                break
            else:
                print("Error: Please select an option from " + str(selection_set))
        except Exception:
            print("Error: Could not convert input to int")

    if choice == 1:
        print("Enter the absolute path of your image:")
        path = input()
        image = load_image(path)
        # try:
        #     image = load_image(path)
        # except Exception:
        #     print("Error: Could not load image")

        feature_vector = resNet50Model().extract_features(image)
        most_similar = find_most_similar(feature_vector)
        print(most_similar['file_name'])
        return True

prompt()