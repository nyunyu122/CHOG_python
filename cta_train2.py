import numpy as np
import scipy.ndimage

import cta_chog
import cta_fspecial
import cta_products

### model パラメタの入った辞書を返す

class Cta_train2():
    def __init__(self, 
                 images=train_imgs,
                 labels=train_orientation_labels,
                 precision='complex128',
                 w_func=['circle',[0,2],[2,2],[4,2]] ,
                 v_sigma=[2.0], 
                 product_options={'monoms':['001']}, 
                 L=5,
                 chog_options=chog_options, # 展開した状態で渡すなら **chog_option とする。最後の引数にする
                 verbosity=1, 
                 output_order=int(0)):  # a gradient is of order 1   
                 
                 

        self.images = images
        self.labels = train_orientation_labels
        self.output_order = output_order
        
        self.precision = precision
        self.w_func = w_func
        print('cta_train, w_func: ' + str(self.w_func))
        self.v_sigma = v_sigma
        self.product_options = product_options
        self.L = L
        self.verbosity = verbosity
        
        self.product_options['output_order']=self.output_order
        
        self.chog_options = chog_options
        
        #self.filter_mode='nearest'  #'replicate' 
        self.filter_mode='wrap' #'circular' 

        self.regu=0.00001 

        # for k = 1:2:numel(varargin),
        #         eval(sprintf('%s=varargin{k+1} ',varargin{k}))
        # %varargin{k+1} が数値精度指定であれば、 varargin{k} を出力する。 おそらく precision の確認
        # end 


        complex_derivatives=np.array([[0,0,(-1j)/8,0,0],
                                      [0,0,1j,0,0],
                                      [(-1)/8,1 ,0,-1, (1)/8],
                                      [0,0,-1j,0,0],
                                      [0,0,(1j)/8,0,0]])

        self.complex_derivatives=np.conj(complex_derivatives)

    
    def cta_train2(self):

        model = {'v_sigma':self.v_sigma,
                 'product_options':self.product_options,
                 'L':self.L,
                 'filter_mode':self.filter_mode,
                 'chog_options':self.chog_options,
                 'output_order':self.output_order}
        

        FIcol = []  ## features
        licol = []  ## targets
        nVcol = 0   ## number of voxels per image

#         print('self.verbosity: ' + str(self.verbosity))
#         print('range(len(self.images)): ' + str(range(len(self.images))))

        flag_FIcol = 0     
        for nI in range(len(self.images)):
            if self.verbosity>0:
                nI_ = nI+1
                print('computing features for image %d' % nI_ ) 

            img = self.images[nI]
            
            label = self.labels[nI]
            
            if self.output_order>0:
                label=label**self.output_order

            shape=img.shape

#             print('chog options')
#             print(self.chog_options)

            chog=Cta_chog(img, self.L, self.precision, self.chog_options).cta_chog() 
            print('cta_chog --done')
            
            if nI==0:
                product_mat=Cta_products(chog,self.product_options, self.output_order).cta_products()
                print('cta_products was called')
                
#                 np.savetxt('product_mat.csv', product_mat)
            #     np.set_printoptions(edgeitems=3, threshold=3000)
            #     print("product_mat")
            #     print(product_mat)
            num_products=product_mat.shape[0]

            num_voting_finc=len(self.v_sigma)

            fimage=np.zeros([shape[0]*shape[1],num_voting_finc*num_products], dtype = self.precision) 

            count=0
            ft_kernel = []
            for vf in range(num_voting_finc):
                ft_kernel.append(np.fft.fftn(Cta_fspecial(shape,'gauss',self.v_sigma[vf],False,self.precision).cta_fspecial()))

                for p in range(num_products):
                    product=product_mat[p,:]
                    if product[2] != 0:
                        A=np.conj(np.squeeze(chog[int(product[0]-1)]["data"][int(product[1])]))
                    else:
                        A=np.squeeze(chog[int(product[0]-1)]["data"][int(product[1])])

                    if product[3]==-1: # 2つ目以降の window 関数がない場合
                            if self.verbosity>1:
                                print('(%d) [%d]%d -> %d' % (product[0],(-1)**product[2],product[1],product[1]))

                            tmp=A
                            
                    else:
                        if product[5] != 0:
                            B=np.conj(np.squeeze(chog[int(product[3]-1)]["data"][int(product[4])]))
                        else:
                            B=np.squeeze(chog[int(product[3]-1)]["data"][int(product[4])])

                        if product[6]==-1: # 3つ目以降の window 関数がない場合        
                                if self.verbosity>1:
                                    print('(%d) [%d]%d x (%d) [%d]%d -> %d' % (product[0],(-1)^product[2],product[1],product[3],(-1)^product[5],product[4],product[9]))

                                tmp=A*B 

                        else:
                            if product[8] != 0:
                                C=np.conj(np.squeeze(chog[int(product[6]-1)]["data"][int(product[7])])) 
                            else:
                                C=np.squeeze(chog[int(product[6]-1)]["data"][int(product[7])])

                            if self.verbosity>1:
                                print('(%d) [%d]%d x (%d) [%d]%d x (%d) [%d]%d -> %d',product[0],(-1)^product[2],product[1],product[3],(-1)^product[5],product[4],product[6],(-1)^product[8],product[7],product[9]) 

                            tmp=A*B*C 

                    l=product[-2]

                    while l>self.output_order:
                        l=l-1 
                        tmp_x_real=scipy.ndimage.correlate(np.real(tmp), np.real(self.complex_derivatives), mode=self.filter_mode)
                        tmp_y_real=scipy.ndimage.correlate(np.real(tmp) , np.imag(self.complex_derivatives), mode=self.filter_mode)
                        tmp_x_imag=scipy.ndimage.correlate(np.imag(tmp), np.real(self.complex_derivatives), mode=self.filter_mode)
                        tmp_y_imag=scipy.ndimage.correlate(np.imag(tmp) , np.imag(self.complex_derivatives), mode=self.filter_mode)
                        tmp= tmp_x_real+ 1j*tmp_y_real + 1j*tmp_x_imag - tmp_y_imag

                    for vf in range(num_voting_finc):
                        tmp=np.fft.ifftn(np.fft.fftn(tmp)*ft_kernel[vf]) 

                        Mask=np.zeros(shape)
                        border=int(np.ceil(np.max(self.v_sigma)/2))
                        Mask[border-1:Mask.shape[0]-border+1, border-1:Mask.shape[1]-border+1]=1
                        tmp[Mask==0]=0

                        test = tmp.reshape(-1,order='F')
                        fimage[:,vf*num_products+count]=tmp.reshape(-1,order='F')
                        
                    count=count+1
            
#             np.savetxt('fimage'+str(nI)+'.csv', fimage)
            
            if flag_FIcol == 0: #FIcol == []:
                FIcol = fimage
                                
                licol = label.reshape(-1, order = 'F')
                flag_FIcol = 1
            else:
                FIcol = np.concatenate([FIcol,fimage])
#               
                
                licol = np.concatenate([licol,label.reshape(-1, order = 'F')]) #20201204ここ! order = 'F' がないのが原因？, axis = 1)
            nVcol = nVcol + shape[0]*shape[1]

        if self.verbosity>0:
            print('solving regression problem ... ') 
            
#         np.savetxt('FIcol.csv', FIcol)
        
        renormfac = np.transpose(np.conj(np.sqrt(np.sum(np.abs(FIcol)**2,axis=0)/nVcol)))
        np.savetxt('renormfac.csv', renormfac)
        invrenormfac = 1/renormfac
        
        FIcol = FIcol * np.conj(np.tile(invrenormfac,(nVcol,1)))         
        Corr = np.dot(np.transpose(np.conj(FIcol)), FIcol)
        b = np.dot(np.transpose(np.conj(FIcol)), licol)
        
        alpha = np.dot( np.linalg.inv( Corr + np.dot(self.regu, np.diag(invrenormfac**2))), b)
        ### np.diag は 入力のベクトルを対角成分に持つ正方行列を作る
        alpha = alpha / np.dot(np.transpose(np.conj(alpha)), b) * np.sum(np.abs(licol))
        ### (alpha'*b) はスカラー
        ### sum(abs(licol(:))) もスカラー
        
#         np.savetxt('alpha.csv', alpha)

        nvf_coeff = alpha.size / num_voting_finc
        ### [np.array()].size は全要素数を返す

        nalpha = invrenormfac * alpha

        model_alpha = []

        for vf in range(num_voting_finc):
            model_alpha.append(nalpha.ravel()[int(vf*nvf_coeff) : int((vf+1)*nvf_coeff)].reshape(alpha.shape))

        model['alpha']=model_alpha
        model['products']=product_mat

        if self.verbosity>0:
            print('done') 

        print('self.verbosity: ' +str(self.verbosity))
        print('self.output_order: ' + str(self.output_order))

        ########## 
        if fimage.any() and (model!=[]) :
            nargout = 2
        else:
            nargout = 1
            print("Warning: nargout != 2")
        ###########

        print('nargout: ' + str(nargout))

        if (self.verbosity>0) or (nargout>1):
            current=0
        #     current=1 
            ## 以降のfigureは self.verbosity>0の場合じゃないとプロットされない
            if self.verbosity>0:   
                plt.figure(4567+self.output_order)

            fimages = []

            for a in range(len(self.images)):
                shape=self.images[a].shape

                if nargout>1:
                    fimages.append(FIcol[current:current+self.images[a].size, :].reshape((shape[0],shape[1], FIcol.shape[1])))

                if self.verbosity>0:
                ### self.verbosity は cta_train2 内では 0 ならプロットなし、1 ならプロットあり。
                
                    if self.output_order ==0:
                        print("self.verbosity >0:, self.output_order ==0:")
                        result = np.real(np.dot(FIcol[current:current+self.images[a].size, :], alpha).reshape(shape, order='F'))

                        plt.figure(figsize=(5,5))
                        plt.imshow(result*(result>0), origin='lower')
                        plt.xlabel('self detection: training image ' + str(a)) 
                        plt.show()

                        ### 20201208 debug.
                        ## image の数だけ回す for loop を一番内側にしないとちゃんとプロットができない
                        ## image の数だけ [1x len(image)] の subplots の枠ができてしまう
#                         if len(self.images)==1:                        
#                             plt.imshow(result*(result>0), origin='lower')

#                         else:
#                             fig, axes = plt.subplots(1, len(self.images))
#                             ### subplots は複数枚のグラフがないと添字で位置を指定できないので、fimage の数による分岐が必要
#                             ### 20201029 ここのプロットのところちょっとおかしい。直す
#                             axes[a].imshow(result*(result>0), origin='lower')
#                             axes[a].set_xticks([], [])
#                             axes[a].set_yticks([], [])
#                         plt.xlabel('self detection: training image ' + str(a)) 
                        
                    else:
                        result=np.dot(FIcol[current:current+self.images[a].size, :], alpha).reshape(shape, order='F')
                        result=result**(1/self.output_order)


                        plt.figure(figsize=(10,10))
                        plt.imshow(self.images[a], cmap='gray', origin='lower')
                        [X,Y]=np.meshgrid(np.arange(1,result.shape[1],5), np.arange(1,result.shape[0],5))
                        plt.quiver(X,Y,np.real(result[0::5,0::5]), np.imag(result[0::5,0::5]), color='red', scale=50) 
                        plt.xlabel('self detection: training image ' + str(a+1))
                        plt.show()
                else:
                    pass

                ## 20201207 debug. 
                # subplots にしたい時は for a in range(len(image)) のloopを一番内側に
#                     if len(self.images)==1:
#                         plt.figure(figsize=(10,10))
#                         plt.imshow(self.images[0], cmap='gray', origin='lower')
#                         [X,Y]=np.meshgrid(np.arange(1,result.shape[1],5), np.arange(1,result.shape[0],5))
#                         plt.quiver(X,Y,np.real(result[0::5,0::5]), np.imag(result[0::5,0::5]), color='red', scale=50)

#                     else:
#                         fig, axes = plt.subplots(len(self.images),1)

#                         [X,Y]=np.meshgrid(np.arange(1,np.ceil(shape[1],2), np.arange(1,shape[0],2)))
#                         for a in range(len(self.images)):
#                             axes[a, 1].imshow(self.images[a], cmap='gray', origin='lower')
#                             axes[a, 1].quiver(X,Y,np.real(result[0::2,0::2]), np.imag(result[0::2,0::2]), color='red', scale=20)

# #                                 axes[0, a].quiver(X,Y,np.real(result[0::2,0::2]), np.imag(result[0::2,0::2]), color='red', scale=20)
# #                                 axes.set_xticks([], [])
# #                                 axes.set_yticks([], [])

#                     plt.xlabel('self detection: training image ' + str(a)) 
#                     plt.show()

                current = current + self.images[a].size
        return [model, fimages]