from PIL import Image
import os


def get_imlist(path, extension):
    '''Returns a list of file names for given extension in a directory'''
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(extension)]

def imresize(im, sz):
    '''Resize an image array using PIL'''
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im, nbins = 256):
    '''Histogram equalization of a greyscale image'''

    # Get histogram
    imhist, bins = histogram(im.flatten(), nbins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1] # normalize

    # linear interpolation of cdf to find new px vals
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf
    
def compute_average(imlist):
    '''Compute the average of a list of images'''

    # Open first image and make into array of floats
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print imname + ' ...skipped'
    averageim /= len(imlist)

    # return avg as uint8
    return array(averageim, 'uint8')

def pca(X):
    # Get dims
    ndata, dim = X.shape
    
    # center
    mean_X = X.mean(axis = 0)
    X = X - mean_X
    
    if dim > ndata:
        M = dot(X, X.T)        # cov matrix
        e, EV = linalg.eigh(M) # eigenval/eigenvec
        tmp = dot(X.T, EV).T
        V = tmp[::-1]          # reverse so biggest vals are first
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # SVD method
        U, S, V = linalg.svd(X)
        V = V[:ndata]
        
    return V, S, mean_X
