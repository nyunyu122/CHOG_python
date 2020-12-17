import numpy as np
from skimage import io,color

class Cta_absorption():
    def __init__(self, img, absorb, noise):
        self.img=img
        self.absorb=absorb
        self.noise=noise

    def cta_absorption(self):
        shape=self.img.shape
        
        texture=color.rgb2gray(color.rgba2rgb(io.imread('./demo5/texture1.png')))

        ## 通常はこれで良いが、octave の出力と合わせるためデバッグ中は下記を使う
        ## octave のスクリプトでは rgb2gray 関数の出力値が int に丸められてしまっているせいで skimage の結果と少し値が違う
#         texture = io.imread('texture1.png')
#         texture = 0.2989*texture[:,:,0] + 0.5870*texture[:,:,1] + 0.1140*texture[:,:,2]
#         texture = texture.astype(int)/255
        
        img=0.7*self.img+0.3*texture[:shape[0],:shape[1]]

        [X, Y] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        Y = Y.transpose()
        Y=Y/shape[1]
        
        img=img*np.exp(-Y*self.absorb)
        
        max_value=self.img.max()

        img=img+self.noise*max_value*np.random.randn(shape[0],shape[1])
        
        return img
        