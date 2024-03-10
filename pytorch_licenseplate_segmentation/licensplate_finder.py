import torch
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("../")
sys.path.append("pytorch_licenseplate_segmentation")
from model import create_model
import utils
# %load_ext autoreload
# %autoreload 2
class Licensplate_finder():
    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model = create_model(aux_loss=True)
        # checkpoint = torch.load('model_v2.pth', map_location='cpu')
        checkpoint = torch.load(os.path.join(os.getcwd(),'pytorch_licenseplate_segmentation/model_v2.pth'), map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        _ = self.model.eval()

        if torch.cuda.is_available():
            self.model.to('cuda')

    def plot(self,img, pred, threshold=0.5):
        plt.figure(figsize=(20, 20));
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(pred.cpu().numpy()[0] > threshold)
        plt.title('Segmentation Output')
        plt.axis('off')

    def pred(self,image, model):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
            return output

    def postprocess_prediction(self,outputs: torch.tensor):
        img = outputs.detach().cpu().numpy()[0]
        img[img > 0] = 1
        img[img <= 0] = 0
        img = (img * 255).astype('uint8')
        return img

    def find_contour_minAreaRect(self,image):
        # Находим контуры на изображении
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = [contour for contour in contours if
                   cv2.contourArea(contour) == max([cv2.contourArea(contour) for contour in contours])][0]

        # Вычисляем ограничивающий прямоугольник для контура
        rect = cv2.minAreaRect(contour)
        # Извлекаем угол наклона прямоугольника
        angle = rect[2]
        return rect

    def subimage(self,image, center, theta, width, height):
        """
        rotate image to set licence plate to horizontal position
        """
        shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)
        matrix = cv2.getRotationMatrix2D(center=center, angle=-theta, scale=1)
        image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)
        image = image[y:y + height, x:x + width]
        return image

    def process(self,image_path,save_to=None):

        image = Image.open(image_path).convert('RGB')
        image_original = np.asarray(image)
        outputs = self.pred(image=image, model=self.model)
        # self.plot(image, outputs, threshold=0.1)
        img = self.postprocess_prediction(outputs)
        rect = self.find_contour_minAreaRect(img)
        center = [int(i) for i in rect[0]]
        theta = (90 - rect[2])
        # theta=0
        width = int(rect[1][1])
        height = int(rect[1][0])
        image = self.subimage(image_original, center, theta, width, height)
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            matplotlib.image.imsave(os.path.join(save_to,os.path.split(image_path)[-1]),image)
        return image,image_original


if __name__ == '__main__':
    licensplate_finder = Licensplate_finder()
    dir = r'/media/yuri/A7/pycharm_projects/Tutorials/licence_plates/images/'
    save_to = r'/media/yuri/A7/pycharm_projects/Tutorials/licence_plates/result/'
    for i, file in enumerate(os.scandir(dir)):
        licensplate_finder.process(file.path,save_to)