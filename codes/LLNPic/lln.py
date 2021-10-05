from PIL import Image, ImageOps
import os   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot 
import imageio

fontsize= 14
ticksize = 14
figsize = (7, 4.0)
params = {"text.usetex": True,
    'font.family':'serif',
    "figure.figsize":figsize, 
    'figure.dpi': 80,
    'figure.edgecolor': 'k',
    'font.size': fontsize, 
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': ticksize,
    'ytick.labelsize': ticksize
}
plt.rcParams.update(params)

img  = Image.open('codes\LLNPic\pics\.jpg') #change the path of the file here 
img_gr = ImageOps.grayscale(img)
data = np.asarray(img_gr)
