import cv2
import numpy as np
import os
import train_model
from PIL import Image



def model_predict(img1):
    
    model = train_model.model()

    test_data = cv2.resize(cv2.imread(img1), (100, 100))

    cv2.imwrite(img1 + '_resized.jpg', test_data)
    np.shape(test_data)
    model_out = model.predict([test_data])

    x = np.reshape(model_out, (100, 100, 3))

    cv2.imwrite(img1 + '_output.jpg', x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="name of image file")

    args = parser.parse_args()
    path = args.path
    model_predict(path)

    