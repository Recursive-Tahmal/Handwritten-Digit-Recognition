import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('Test.model')

# load the data
image_number = 0
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        img = cv.imread(f"digits/{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(
            f"The number is probably a {np.argmax(prediction)} (with {np.max(prediction) * 100} % confidence)")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1

    except Exception as e:
        print("Error: ", e)


print("Done!")
