from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from fastai.vision import *

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_learner("densenet161_rus/")
        
    def predict(self, img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # Не забудьте перенести все трансофрмации, которые вы использовали при тренировке
        # Для этого будет удобно сохранить питоновский объект с ними в виде файла с помощью pickle,
        # а потом загрузить здесь.

        # Обработка картинки сейчас производится в методе process image, а здесь мы должны уже применить нашу
        # модель и вернуть вектор предсказаний для нашей картинки

        return self.model.predict(self.process_image(img_stream))[0]
    
    def predict_proba(self, img_stream):
        #Этот метод возвращает вероятность предсказанного класса
        
        probs = (self.model.predict(self.process_image(img_stream))[2]).numpy()
        return round(probs[self.model.predict(self.process_image(img_stream))[1]]*100, 1)

    def process_image(self, img_stream):
        # используем open_image, чтобы получить картинку из потока и изменить размер
        image = open_image(img_stream).resize((3, 224, 224))
        return image

