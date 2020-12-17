import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import io, color


class Cta_fspecial:
    
    " training image と同じ大きさ （shape） の kernel を返す "
    " image の大きさ （shape） のみに依存。image の中身は関係ない "

    def __init__(self, shape, kname='gauss', kparams = 1.5, centered = False, precision = 'complex128'):
        self.kname = kname
        self.kparams = kparams
        self.shape = shape

        self.centered = centered
        
        if precision == 'complex128':
            self.precision = 'float64' #cta_fspecial の時点ではデータは全て実数なので、complexである必要がない
            ### 20200908 complex で指定しているのに精度が実数なのは confusing なので、将来的には「double」か「single」の指定に直したい
        elif precision == 'complex64':
            self.precision = 'float32'
    
    def cta_fspecial(self):
        assert len(self.shape)==2, 'Error: an input image is not properly read.'

        [X,Y]=np.meshgrid(np.arange(self.shape[0], dtype=self.precision),np.arange(self.shape[1], dtype=self.precision))
        X = np.transpose(X - math.ceil(self.shape[0]/2))
        Y = np.transpose(Y - math.ceil(self.shape[1]/2))


        if self.kname == 'gauss':
            assert(isinstance(self.kparams, float))
            sigma = 2*(self.kparams**2)
            R2 = (X**2/sigma + Y**2/sigma)
            kernel = np.exp(-R2)
            kernel = kernel/np.sum(kernel)

        elif self.kname == 'circle':
            radius = self.kparams[0]
            radial_smooth = 2*(self.kparams[1]**2) 
            distance_sqr = np.sqrt(X**2 + Y**2)
            R2 = ((distance_sqr-radius)**2)/radial_smooth
            
            kernel = np.exp(-R2)
            kernel = kernel/np.sum(kernel)

        else:
            kernel = None
            print("unsupported kernel. kernel type shoud be 'gauss' or 'cicle'")

        if not self.centered:
            kernel = np.fft.fftshift(kernel)
            
        return kernel



