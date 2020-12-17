# cta_apply2(test_images{l},model_orient,'padding',0,'precision','single') 

import numpy as np
import scipy.ndimage

class Cta_apply2():
    def __init__(self, image, model, padding=0, precision='complex64'):

        self.precision=precision # 'double' 
        self.verbosity=0
        self.padding=padding

        self.image=image
        self.model=model

        self.shape=image.shape 

        # if padding>0
        #     padded_shape=cta_fft_bestshape(shape+padding) 
        #     original_shape=shape 
        #     img=zeros(padded_shape) 
        #     img(1:shape(1),1:shape(2))=image 
        #     image=img 
        # end 

        # shape=size(image) 

        self.complex_derivatives=np.array([[0,0,(-1j)/8,0,0],
                                           [0,0,1j,0,0],
                                           [(-1)/8,1 ,0,-1, (1)/8],
                                           [0,0,-1j,0,0],
                                           [0,0,(1j)/8,0,0]])


        self.complex_derivatives=np.conj(self.complex_derivatives)   

    def cta_apply2(self):
        chog=Cta_chog(self.image, self.model['L'], self.precision, self.model['chog_options']).cta_chog()
        #model[4] = model.chog_options
#         chog=Cta_chog(image, self.L, self.precision, self.chog_options).cta_chog() 

        num_products=self.model['products'].shape[0] #size(model.products,1)

        H=np.zeros(self.shape, dtype=self.precision) # test_image と同じ大きさのゼロ行列

        for vf in range(len(self.model['v_sigma'])): #=1:numel(model.v_sigma),
            L= self.model['products'][0, -2] #model.products(1,end-1)
        #     print('L: ' + str(L))
            H_tmp=np.zeros(self.shape,self.precision) 
            for p in range(num_products):
                product=self.model['products'][p,:] #model.products(p,:)
                if product[2] != 0:
                    A=np.conj(np.squeeze(chog[int(product[0]-1)]['data'][int(product[1])]))
                else:
                    A=np.squeeze(chog[int(product[0]-1)]['data'][int(product[1])])


                if product[3]==-1: # 2つ目以降の window 関数がない場合
                    if self.verbosity>1:
                        print('(%d) [%d]%d -> %d' % (product[0],(-1)**product[2],product[1],product[1]))
                    tmp=A
                else:
                    if product[5] != 0:
                        B=np.conj(np.squeeze(chog[int(product[3]-1)]['data'][int(product[4])]))
                    else:
                        B=np.squeeze(chog[int(product[3]-1)]['data'][int(product[4])])


                    if product[6]==-1: # 3つ目以降の window 関数がない場合        
                        if self.verbosity>1:
                            print('(%d) [%d]%d x (%d) [%d]%d -> %d' % (product[0],(-1)^product[2],product[1],product[3],(-1)^product[5],product[4],product[9]))
                        tmp=A*B 

                    else:
                        if product[8] != 0:
                            C=np.conj(np.squeeze(chog[int(product[6]-1)]['data'][int(product[7])])) 
                        else:
                            C=np.squeeze(chog[int(product[6]-1)]['data'][int(product[7])])

                        if self.verbosity>1:
                            print('(%d) [%d]%d x (%d) [%d]%d x (%d) [%d]%d -> %d',product[0],(-1)^product[2],product[1],product[3],(-1)^product[5],product[4],product[6],(-1)^product[8],product[7],product[9]) 
                        tmp=A*B*C 


                l=product[-2]

                while l<L:
                    L=L-1

        #             H_tmp=imfilter(H_tmp,complex_derivatives,model.filter_mode)
                    H_tmp_x_real=scipy.ndimage.correlate(np.real(H_tmp), np.real(self.complex_derivatives), mode=self.model['filter_mode'])
                    H_tmp_y_real=scipy.ndimage.correlate(np.real(H_tmp) , np.imag(self.complex_derivatives), mode=self.model['filter_mode'])
                    H_tmp_x_imag=scipy.ndimage.correlate(np.imag(H_tmp), np.real(self.complex_derivatives), mode=self.model['filter_mode'])
                    H_tmp_y_imag=scipy.ndimage.correlate(np.imag(H_tmp) , np.imag(self.complex_derivatives), mode=self.model['filter_mode'])
                    H_tmp= H_tmp_x_real+ 1j*H_tmp_y_real + 1j*H_tmp_x_imag - H_tmp_y_imag


                H_tmp = H_tmp + self.model['alpha'][vf][p]*tmp

            while L>self.model['output_order']: #(L>model.output_order)
                L=L-1 
        #         H_tmp=imfilter(H_tmp,complex_derivatives,model.filter_mode) 
                H_tmp_x_real=scipy.ndimage.correlate(np.real(H_tmp), np.real(self.scomplex_derivatives), mode=self.model['filter_mode'])
                H_tmp_y_real=scipy.ndimage.correlate(np.real(H_tmp) , np.imag(self.complex_derivatives), mode=self.model['filter_mode'])
                H_tmp_x_imag=scipy.ndimage.correlate(np.imag(H_tmp), np.real(self.complex_derivatives), mode=self.model['filter_mode'])
                H_tmp_y_imag=scipy.ndimage.correlate(np.imag(H_tmp) , np.imag(self.complex_derivatives), mode=self.model['filter_mode'])
                H_tmp= H_tmp_x_real+ 1j*H_tmp_y_real + 1j*H_tmp_x_imag - H_tmp_y_imag

       #     ft_kernel=fftn(cta_fspecial('gauss',model.v_sigma(vf),shape,false,precision))
            ft_kernel=np.fft.fftn(Cta_fspecial(self.shape, 'gauss', self.model['v_sigma'][vf], False, self.precision).cta_fspecial())

            if self.model['output_order']==0:
                H=H+np.real(np.fft.ifftn(np.fft.fftn(H_tmp)*ft_kernel))
            else:
                H=H+np.fft.ifftn(np.fft.fftn(H_tmp)*ft_kernel) 

        if self.model['output_order']>0: #(model.output_order>0)
        #     H=abs(H).*(H./abs(H)).^(1/model.output_order) 
            H=np.abs(H)*(H/np.abs(H))**(1/self.model['output_order'])

        # if padding>0
        #     H=H(1:original_shape(1),1:original_shape(2)) 

        Mask=np.zeros(H.shape) 
        border=int(np.ceil(np.max(self.model['v_sigma']))) # ceil(max(model.v_sigma)) 
        Mask[border-1:Mask.shape[0]-border+1, border-1:Mask.shape[1]-border+1]=1
        H[Mask==0]=0
        
        return H

