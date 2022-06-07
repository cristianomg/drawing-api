import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image

class CNN():
    def __init__(self, model, labels) -> None:
        self.labels = labels
        self.model = models.load_model(model)

    def predict(self, image):
        Image.open(image).save("img.png")

        img = Image.open('img.png')
        img = np.array(img)[:, :, 3]

        opencvImage = cv2.resize(img, (28, 28))

        predict = self.model.predict(opencvImage.reshape(1, 28, 28))

        print(predict)

        return self.__get_predicted_label(predict[0])

    def __get_predicted_label(self, predict):
        max_predict = predict.max();
        print(max_predict);
        if (predict.max() * 100 > 90):
            return "Seu desenho é um " + self.labels[np.argmax(predict)]
        return 'Não sei qual é o seu desenho tente novamente.'