import numpy as np
import cv2 as cv

from PIL import Image

class Erase(object):
    '''
    Class to erase COR logo from images
    '''
    def __init__(self, i=39,j=0,h=90,w=265,v=0):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.v = v
    
    def __call__(self, img):
        img_array = np.array(img)
        img_array[self.i:self.i+self.h, self.j:self.j+self.w, :] = 0
        return Image.fromarray(img_array)
            
class LowerBrightness(object):
    
    def __init__(self, brigthness_factor:float=0.5):
        self.bf = brigthness_factor
    
    def __call__(self, img:Image):
        image = img.convert("YCbCr")
        y, cb, cr = image.split()
        y = y.point(lambda b: b*self.bf)
        image = Image.merge("YCbCr",(y, cb, cr))
        return image.convert("RGB")

class Normalize(object):
    '''
    Performs histogram normalization
    '''
    def __init__(self, alpha:int=0, beta:int=255):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, img:Image):
        img_array = np.array(img)
        img_norm = np.zeros(img_array.shape)
        img_norm = cv.normalize(img_array, img_norm, self.alpha, self.beta, cv.NORM_MINMAX)
        return Image.fromarray(cv.cvtColor(img_norm, cv.COLOR_GRAY2RGB))