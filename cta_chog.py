# CTA_CHOG computes Circular Fourier HOG features according to eq. (3) in 
#
# Henrik Skibbe and Marco Reisert 
# "Circular Fourier-HOG Features for Rotation Invariant Object Detection in Biomedical Images"
# in Proceedings of the IEEE International Symposium on Biomedical Imaging 2012 (ISBI 2012), Barcelona 
#
# chog=cta_chog(image,...                       # NxM image
#               'w_func',{'circle',[4,2]},...   # window function(s), e.g. {'circle',[0,3],[3,3],[6,3]}
#               'L',5,...                       # maximum angular frequency
#               'precision','double',...        # precision ('double'/'single')
#               'presmooth','1.5',...           # smoothing before computing the gradient
#               'l2',true,...                   # l2-normalization     
#               'gamma',0.8);                   # gamma corection of gradient magnitude
#
# returns a list (one element for each window functions)
#
#     chog{w}.data      # CxNxM image containing the C expansion coefficients        
#     chog{w}.L         # maximum angular frequency
#     chog{w}.shape     # [N,M]
#
#
# Note that these features are NOT rotation invariant
#
# See also cta_invrts cta_train cta_apply


#
# Copyright (c) 2011, Henrik Skibbe and Marco Reisert
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.
#


w_func=['circle',[4,2]]

chog_options={'w_func':w_func, 
              'presmooth':2.0, # initial smoothing (before computing the image gradient)
              'l2':True, # in order to cope with the absorption we choose l2 normalization
              'gamma':0.8, # makes the gradient orientation more dominant over magnitude
              'verbosity':0}


import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.ndimage
from skimage import io, color

# import cta_fspecial # from [.py filename] import [class name]

eps = sys.float_info.epsilon ## __init__ の中に self.eps として書く？ （わざわざinitの中に書かなくてもいい気がする）


class Cta_chog():
    def __init__(self, Image, L=5, precision='complex128', chog_options=chog_options):
        
        self.L=L;
    
        self.w_func=chog_options['w_func'];
#         print("chog_options['w_func']" +str(chog_options['w_func']))
        print('cta_chog, w_func: ' + str(self.w_func))
        self.presmooth=chog_options['presmooth'];
        self.l2=chog_options['l2'];
        
        self.gamma=chog_options['gamma']
        self.verbosity=chog_options['verbosity'];

        #filter_mode='nearest' # matlab の replicate モードに相当
        self.filter_mode='wrap' # matlab の circular モードに相当

        self.precision=precision; # 'double'

        self.complex_derivatives=np.array([[0,0,(-1j)/8,0,0],
                                      [0,0,1j,0,0],
                                      [(-1)/8,1 ,0,-1, (1)/8],
                                      [0,0,-1j,0,0],
                                      [0,0,(1j)/8,0,0]])
        
        self.shape=Image.shape

        self.image = Image

        
    def cta_chog(self):        

        if self.presmooth > 0:
            
            ft_kernel=np.fft.fftn(Cta_fspecial(self.shape, 'gauss', self.presmooth, False,
                                               self.precision).cta_fspecial());
            image=np.fft.ifftn(np.fft.fftn(self.image)*ft_kernel);
            
#             plt.imshow(np.abs(image), cmap='gray')
#             plt.show()

            # computing the complex derivatives df/dx+idf/dy (paper text below eq. (4))
            #     gradient_image=imfilter(image,complex_derivatives,filter_mode);
            GI_x_real=scipy.ndimage.correlate(np.real(image), np.real(self.complex_derivatives), mode=self.filter_mode)
            GI_y_real=scipy.ndimage.correlate(np.real(image) , np.imag(self.complex_derivatives), mode=self.filter_mode)
            GI_x_imag=scipy.ndimage.correlate(np.imag(image), np.real(self.complex_derivatives), mode=self.filter_mode)
            GI_y_imag=scipy.ndimage.correlate(np.imag(image) , np.real(self.complex_derivatives), mode=self.filter_mode)

            gradient_image= GI_x_real+ 1j*GI_y_real + 1j*GI_x_imag - GI_y_imag
                        

            # computing the gradient magnitude
            gradient_magnitude = np.abs(gradient_image)
           
            inv_gradient_magnitude=1/(gradient_magnitude+eps)
            ## ゼロで割る事態を避けるために、分母に小さい値（倍精度小数点の精度 eps）を足している

            # gamma correction (paper eq. (4))
            if self.gamma!=1:
                gradient_magnitude=gradient_magnitude**self.gamma

            # computing gradient orientation ^g (paper text below eq. (1))
            gradient_direction=gradient_image*inv_gradient_magnitude

            Fourier_coefficients = np.zeros((self.L+1, self.shape[0], self.shape[1]),
                                            dtype = self.precision)
            
            # iterative computation of the coefficients e^l(^g) (paper eq. (3))
            Fourier_coefficients[0,:,:]=gradient_magnitude
            Fourier_coefficients[1,:,:]=gradient_direction*gradient_magnitude.astype(self.precision);
            
            current=gradient_direction;
            for l in range(2,self.L+1):
                current=current*gradient_direction;
                Fourier_coefficients[l,:,:]=current*gradient_magnitude;

            chog = list(np.zeros(len(self.w_func)-1))

            # computung a^l_w(x) : convoluion with window function(s) (paper eq. (3))

            if len(self.w_func)>0:
                for w in range(len(self.w_func)-1):
                    
                    if self.l2:
                        tmp2 = np.zeros(self.shape);
                        
#                     print('w_func[w+1]')
#                     print(w_func[w+1])
                    chog_w={'data':np.zeros((self.L+1,self.shape[0],self.shape[1]),
                                           dtype = self.precision),
                           'L':self.L,
                           'shape':self.shape,
                           'w_func':w_func[0],
                           'w_param':w_func[w+1]}

                    chog[w] = chog_w
                    
                    if self.verbosity > 0:
                        wf=Cta_fspecial(self.shape, self.w_func[0], self.w_func[w+1], True,
                                        self.precision).cta_fspecial()
                        

                    wf=Cta_fspecial(self.shape, self.w_func[0], self.w_func[w+1], False,
                                                 self.precision).cta_fspecial()
                    ft_kernel = np.fft.fftn(wf)
                    
                    for l in range(self.L+1):
                        tmp=np.fft.ifftn(np.fft.fftn(Fourier_coefficients[l,:,:])*ft_kernel)
                        chog[w]['data'][l]=tmp
                        
                        if self.l2:
                            tmp2 = tmp2 + np.real(chog[w]['data'][l])**2+np.imag(chog[w]['data'][l])**2
                            ## l=0 の値から for 文で計算はするが、最終的に l=L 時の値しか使わない
                    
                    if self.l2:
                        tmp2=np.sqrt(tmp2)+eps                        
                        chog[w]['data'] = chog[w]['data']/np.tile(tmp2.reshape([1, tmp2.shape[0], tmp2.shape[1]]),(self.L+1,1,1))
            
                
            else:# w_funcの要素が空のとき
                chog[0]=Fourier_coefficients
                chog[1]=self.L
                chog[2]=self.shape
            
        return chog
    