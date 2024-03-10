import torch
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import numpy as np
import re
import pandas as pd
from PIL import Image

from torchvision import transforms
import easyocr
from pprint import pprint
from skimage.io import imread,imsave
from pytorch_licenseplate_segmentation.licensplate_finder import Licensplate_finder

class Licensplate_region_identifier():
    def __init__(self):
        self.licensplate_finder = Licensplate_finder()
        self.reader = easyocr.Reader(['ru','en'])
        self.df = pd.read_excel(r'numbers.xlsx', index_col=False)
        self.df['list_numbers'] = self.df['numbers'].apply(lambda x: [int(i) for i in str(x).split(',')])

    def crop_registration_region(self,img, ratio=0.3):
        """
        crop initial image by licence plate coords predicted by Licensplate_finder
        """
        len_to_cut = int(img.shape[1] * (1 - ratio))
        img = img[:, len_to_cut:, :]
        return img
    def postprocess_results(self,results):
        results = [results[i][1] for i in range(len(results))]
        results=[re.sub("[^0-9]", "", result) for result in results] # remove all not digits
        # for i,result in enumerate(results):
        #     if len(result) > 3:
        #         results[i]=result[-3:]
        return results


    def process(self,image_path):
        image_cropped, image_original = self.licensplate_finder.process(image_path)
        image_cropped = self.crop_registration_region(image_cropped)
        results = self.reader.readtext(image_cropped)
        results = self.postprocess_results(results)
        for region_number in results:
            regions = self.df[self.df['list_numbers'].apply(lambda x: int(region_number) in x )]['region']
            if len(regions) > 0:
                return regions.tolist()[0]
        for region_number in results: # вертикальная линия слева от номера региона иногда распознаётся как число 1.
            # Поэтому, если номера из списка номеров не обнаружено, то рассматриваем номера, начинающиеся с 1
            if len(region_number) > 1 and region_number[0] == '1':
                region_number=region_number[1:]
                regions = self.df[self.df['list_numbers'].apply(lambda x: int(region_number) in x)]['region']
                if len(regions) > 0:
                    return regions.tolist()[0]
        return None # если номер не найден

if __name__ == '__main__':
    licensplate_region_identifier = Licensplate_region_identifier()
    # image_path = r'/home/yuri/Downloads/e.png'
    image_path = r'/media/yuri/A7/pycharm_projects/Tutorials/licence_plates/images/R6ih80R17Yk.jpg'
    # image_path = r'/media/yuri/A7/pycharm_projects/Tutorials/licence_plates/images/1520287861170497313.jpg'
    result = licensplate_region_identifier.process(image_path)
    pprint(result)

