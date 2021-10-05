from PIL import Image, ImageOps
import os   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot 
import imageio

fontsize= 16
ticksize = 14
figsize = (6, 3)
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

img  = Image.open('codes\LLNPic\pics\ms.jpg') #change the path of the file here 
img_gr = ImageOps.grayscale(img)
data = np.asarray(img_gr)


def corrupt(data,p_bernoulli):
    x_pix = data.shape[0]
    y_pix = data.shape[1]
    
    noise_arr_ver = np.random.binomial(1, p_bernoulli, size=y_pix)
    noise_ver =np.tile(noise_arr_ver, (x_pix, 1))

    noise_arr_hor = np.random.binomial(1, p_bernoulli, size=x_pix)
    noise_hor = np.transpose([noise_arr_hor] * y_pix)
    noise = np.multiply(noise_hor,noise_ver)
    
    data_corrupt = np.multiply(data,noise)-p_bernoulli
    return data_corrupt

def average(data,T,p_bernoulli):
    ave =  np.full_like(data,0)
    for i in range(1,T+1):
        ave = corrupt(data = data,p_bernoulli = p_bernoulli) + ave
    return ave/T    

low_bounds = [1,100,200,300,400,500,600,700,800,900,1000]
up_bounds = [100,199,299,399,490,599,699,799,899,999,5001]
intervals = [1,10,20,30,40,50,60,70,80,90,50]

mesh = []
for i in range(len(low_bounds)):
    for j in range(low_bounds[i],up_bounds[i],intervals[i]):
        mesh.append(j)

filenames = []
for t in mesh:
    ber = 0.5
    X_ave = average(data = data , T = t, p_bernoulli = ber)
    X = corrupt(data,p_bernoulli = ber)
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='gray')
    plt.title(r"$X_i$", pad=10)
    #plt.title(r"$i$"'= {}'.format(t), loc='right', pad=10)

    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(X_ave, cmap='gray')
    plt.title(r"$\frac{1}{N} \sum_{i=1}^N X_i$", loc='left', pad=10)
    plt.title(r"$N$"'= {}'.format(t), loc='right', pad=10)

    plt.axis('off')
    plt.tight_layout()
    filename = 'codes\LLNPic\pics\pic_'+f'{t}.jpg'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()# build gif
with imageio.get_writer('codes\LLNPic\ms_lln.mp4', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)



