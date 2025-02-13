import mimetypes
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import io
from PIL import Image
import os
from datetime import datetime
from pytz import timezone

def current_time():
    return datetime.now(timezone("Brazil/East")).strftime("%Y-%m-%d_%H-%M-%S")

def get_file_extension(filename:str):
    return os.path.splitext(filename)[1]

def get_graph_file(image, title:str=''):
    plt.plot(image)
    buf = io.BytesIO()
    plt.title(title)
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    return cv.UMat(np.array(Image.open(buf), dtype=np.uint8)) #  Needed to convert PIL format to UMat

def get_filenames_from_folder(folder:str):
    '''
    Returns all filenames in a directory tree
    '''
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root,file).replace('\\','/'))
    return file_list

def get_folders_from_dir(dir):
    # https://stackoverflow.com/questions/49882682/how-do-i-list-folder-in-directory
    filenames= os.listdir (dir) # get all files' and folders' names in the current directory
    result = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(dir), filename)): # check whether the current object is a folder or not
            result.append(filename)
    return result
    
def remove_path(filename):
    return os.path.basename(filename)

def remove_extension(filename):
    return os.path.splitext(filename)[0]