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
    
class UnsharpMask(object):
    def __init__(self, ksize=(0,0), sigmaX=2.0, alpha=2.0,beta=-1.0,gamma=0):
        self.ksize=ksize
        self.sigmaX=sigmaX
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def __call__(self,img:Image):
        img_array = np.array(img)
        gaussian = cv.GaussianBlur(img_array,self.ksize,self.sigmaX)
        unsharp = cv.addWeighted(img_array,self.alpha,gaussian,self.beta, self.gamma)
        return Image.fromarray(unsharp)
        
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
        img_norm = cv.normalize(img_array, None, self.alpha, self.beta, cv.NORM_MINMAX)
        # img_rgb = cv.cvtColor(img_norm, cv.COLOR_GRAY2RGB)
        return Image.fromarray(img_norm)
    
class CLAHE(object):
    '''
    '''
    def __init__(self, clip_limit=1, tile_size=(5,5)):
        self.cl = clip_limit
        self.ts = tile_size
    
    def __call__(self, img:Image):
        img_array = np.array(img)
        img_gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        clahe = cv.createCLAHE(clipLimit=self.cl, tileGridSize=self.ts)
        res = clahe.apply(img_gray)
        res = cv.cvtColor(res, cv.COLOR_GRAY2RGB)
        return Image.fromarray(res)

class Threshold(object):
    '''
    '''
    def __init__(self, max_val=200, thresh=255):
        self.max_val = max_val
        self.thresh = thresh
        
    def __call__(self, img:Image):
        img_array = np.array(img)
        gray_image = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        _, binary_mask = cv.threshold(gray_image, self.max_val, self.thresh, cv.THRESH_BINARY)
        result_image = cv.bitwise_not(img_array, img_array, mask=binary_mask)
        return Image.fromarray(result_image)

class Equalize(object):
    '''
    Performs histogram equalization
    '''
    def __call__(self, img):
        img_array = np.array(img)
        image_ycrcb = cv.cvtColor(img_array, cv.COLOR_RGB2YCrCb)
        image_ycrcb[0] = cv.equalizeHist(image_ycrcb[0])
        res = cv.cvtColor(image_ycrcb, cv.COLOR_YCrCb2RGB)
        return Image.fromarray(res)