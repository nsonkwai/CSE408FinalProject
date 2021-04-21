from skimage.util import img_as_ubyte
from skimage.color import rgb2hsv, rgba2rgb
from imageio import imread
from skimage import io
from matplotlib import pyplot as plt
from numpy import array, r_
import numpy as np

# util to show images
def showImage(im):
    io.imshow(im)
    plt.show()

# helper to do fspecial
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def bpdhe(im):
    print('image : \n', im)
    im = img_as_ubyte(im)

    hsv = array(rgb2hsv(im))

    print('hsv : \n', hsv)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # correct this puppy
    i = hsv[:, :, 2] * 255

    print('h : \n', h)
    print('s : \n', s)
    print('i : \n', i)

    ma = max(i.astype(int).flatten())
    mi = min(i.astype(int).flatten())

    bins = int((ma - mi) + 1)
    print('number of bins : ', bins)


    plt.hist(i.flatten().astype(float), bins, density=1, color='green', alpha=0.7)
    plt.show()

    
    # print(bins)
    pass


def main():
    im = imread('1x3_red_border.png')
    im = rgba2rgb(im)
    bpdhe(im)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()