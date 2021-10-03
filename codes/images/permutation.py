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

img  = Image.open('codes\images\pics\picasso.jpg') #change the path of the file here 
img_gr = ImageOps.grayscale(img)
data = np.asarray(img_gr)

def square(data): # turning the data into a M*M matrix
    x_pix = data.shape[0]
    y_pix = data.shape[1]
    if x_pix>y_pix:
        return(data[0:y_pix,0:y_pix]) 
    else:
        return(data[0:x_pix,0:x_pix])


def single_per_matrix(N): #π
    per = np.identity(N)
    i = np.random.randint(N)
    j = np.random.randint(N)
    temp_i = np.copy(per[:,i])
    temp_j = np.copy(per[:,j]) 
    per[:,i] = temp_j
    per[:,j] = temp_i 
    return per


def multiple_per_matrix(T,N): #π_1 o π_2 o ... o π_T
    mpm =  np.identity(N)
    for t in range(T):
        mpm = mpm@single_per_matrix(N)
    return mpm    

def rand_per(d,T): #(π_1 o π_2 o ... o π_T)' @ d @(π_1 o π_2 o ... o π_T)
    N = d.shape[0]
    per_m = multiple_per_matrix(T,N)
    test = np.matrix.transpose(per_m)@d@per_m
    return test

#making the video
data_sq = square(data)
filenames = []
for t in range(201):
    per_data = rand_per(data_sq , T = t)
    plt.subplot(1, 2, 1)
    plt.imshow(data_sq, cmap='gray')
    plt.title(r"$D$", pad = 10)

    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(per_data, cmap='gray')
    plt.title(r"$\big(\Pi_{i=1}^T \pi_i\big)^\top D ~\big(\Pi_{i=1}^T \pi_i\big)$", loc='left',pad=10)
    plt.title(r"$T$"'= {}'.format(t), loc='right',pad=10)

    plt.axis('off')
    plt.tight_layout()
    filename = 'codes\images\pics\pic_'+f'{t}.jpg'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()# build gif
with imageio.get_writer('codes\images\picasso_perm.mp4', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)