import numpy as np
import pprint

import cta_fspecial
import cta_chog

class Cta_products():
    def __init__(self, Fourier_coefficients, product_options, output_order=0):
        self.monoms = product_options['monoms']
        self.feature_order=[0,5];
        self.angular_power=[self.feature_order[0],self.feature_order[1]];
        self.angular_cross=[1,2];
        self.Fourier_coefficients = Fourier_coefficients
        self.num_features=len(self.Fourier_coefficients) # window の数
        
        self.feature_order[0]=max([self.feature_order[0], output_order])
        
    def in_interval(self, value, interval): #value が1つの値の場合でもリストで与える。
        ok=True
        if len(interval)<2:
            ok=False
        for v in range(len(value)):
            if (value[v]<interval[0] or value[v]>interval[1]):
                ok=False
        return ok

### l=0 がなかったのはここだーーーーー！！！！！
# if exist('output_order'),
#     feature_order(1)=max(feature_order(1),output_order);
# end;

# assert(max(cellfun(@numel,monoms))<4);
# assert(min(cellfun(@numel,monoms))>0);
###
    
    def cta_products(self):
        product_mat=[]
        test = []
        for m in range(len(self.monoms)):
            morder=len(self.monoms[m]) #monoms{m} = '01' なら morder = 2
            #print(morder)
            monom=self.monoms[m]
            #print('monom: ' + str(monom))
            for a in range(1, self.num_features+1): # a は window function の番号を表す
                #print('a: ' + str(a))
                for la in range(self.Fourier_coefficients[a-1]["L"]+1): # Fourier_coefficients[a][1] には L が入っている
#                     print('la: ' +str(la))
                    if morder==1: #window function を1種類のみ使う場合
                        assert float(monom[0])==0, 'ERROR: conj is not necessary' #条件式が False の場合にエラーを投げる
                        new_product=[a,la,float(monom[0]),
                                    -1,-1,-1,
                                    -1,-1,-1,
                                    la,0] # +１で本当に合ってるか確認する
                        product_mat.append(new_product)
        #         #print(product_mat)

                    else:
                        for b in range(a,self.num_features+1):
#                             print('b: ' + str(b))
                            start_bl=0;
                            if a==b and monom[0]==monom[1]:
                                start_bl=la;
                            for lb in range(start_bl, self.Fourier_coefficients[b-1]["L"]+1):
#                                 print('lb: ' + str(lb))
                                if morder==2: ## window function を２種類使う場合
                                    l=((-1)**float(monom[0]))*la + ((-1)**float(monom[1]))*lb
                                    #print('-- l: ' + str(l))
        #                             test = (in_interval([l],feature_order) and 
        #                                     ((a==b and in_interval([la,lb],angular_power)) or
        #                                     (a!=b and in_interval([la,lb],angular_cross))))
        #                             #print(test)
                                    if (Cta_products(self.Fourier_coefficients, product_options).in_interval([l],self.feature_order) and ((a==b and Cta_products(self.Fourier_coefficients, product_options).in_interval([la,lb],self.angular_power)) or (a!=b and Cta_products(self.Fourier_coefficients, product_options).in_interval([la,lb],self.angular_cross)))):

                                        new_product=[a,la,float(monom[0]),
                                                     b,lb,float(monom[1]),
                                                     -1,-1,-1,
                                                     l,0];
                                        product_mat.append(new_product)
        #         print(product_mat)
                                
                                ### 20201002 for debug
                                elif morder==3: ## window function を3種類使う場合
#                                     print('morder==3')
#                                     print('self.Fourier_coefficients.shape')
#                                     print(len(self.Fourier_coefficients))

                                    for c in range(b, self.num_features+1):
#                                         print('c: ' + str(c))
#                                         print('self.Fourier_coefficients[c]')
#                                         print(self.Fourier_coefficients[c])
                                        
                                
                                        start_cl=0;
                                        if c==b and monom[1]==monom[2]:
                                            start_cl=lb
                                            
                                        if c==a and monom[0]==monom[2]:
                                            start_cl=np.max(la,start_cl)
#                                         print('start_cl: ' + str(start_cl))
#                                         print('self.Fourier_coefficients[c][1]+1: ' + str(self.Fourier_coefficients[c][1]+1))
                                        for lc in range(start_cl,self.Fourier_coefficients[c-1]["L"]+1):
                                            #print('lc: ' + str(lc))
        #                                     if np.min([la,lb,lc])>0
                                            if np.min([lb,lc])>0:
        #                                     #if 1
                                                l=(((-1)**float(monom[0]))*la 
                                                   +((-1)**float(monom[1]))*lb
                                                   +((-1)**float(monom[2]))*lc)
                                                #print('---l: ' + str(l))
#                                                 print('self.feature_order')
#                                                 print(self.feature_order)
                                                if (Cta_products(self.Fourier_coefficients,product_options).in_interval([l],self.feature_order) and (((a==b and b==c and a==c) and Cta_products(self.Fourier_coefficients, product_options).in_interval([la,lb,lc],self.angular_power)) or ((a!=b or b!=c or a!=c) and Cta_products(self.Fourier_coefficients, product_options).in_interval([la,lb,lc],self.angular_cross)))):

                                                    new_product=[a,la,float(monom[0]),
                                                                 b,lb,float(monom[1]),
                                                                 c,lc,float(monom[2]),
                                                                 l,0];
                                                    product_mat.append(new_product)

        product_mat = np.array(product_mat)
#         product_mat = product_mat[product_mat[:,-2]!=0] # コメントアウト

        
        # v = np.sort(product_mat[:,-2])[::-1]
        indx = np.argsort(product_mat[:,-2])
        product_mat=product_mat[indx]

#         # -2　列目を sort された状態を保ちつつ、さらに 1 列目で sort
#         for i in range(self.feature_order[1]+1):
#             i_indx = np.argsort(product_mat[product_mat[:,-2]==i][:,0])
#             if i == 0:
#                 i_product_mat=product_mat[product_mat[:,-2]==i][i_indx]
#             else:
#                 i_product_mat = np.concatenate([i_product_mat, product_mat[product_mat[:,-2]==i][i_indx]], axis=0)
# #         print(i_product_mat)
#         product_mat = i_product_mat
        
#         # （列, 規則）: (-2,降） → (0,昇) → (1,昇) → (3, 昇)  → （4, 昇） の順に sort
        # 20201005 この部分は octave と出力を揃えるため
        print('self.feature_order[1]')
        print(self.feature_order[1])
        for i in range(self.feature_order[1],-1,-1):#-2
            tmp_i = product_mat[product_mat[:,-2]==i]
#             print('i: ' + str(i))
    
            for j in range(1, self.feature_order[1]+1):#0
#                 print('j: ' + str(j))
                tmp_j = tmp_i[tmp_i[:,0]==j]
#                 print(tmp_j)
                
                for k in range(self.feature_order[1]+1):#1
#                     print('k: ' + str(k))
                    tmp_k = tmp_j[tmp_j[:,1]==k]
#                     print(tmp_k)
                    
                    for l in range(self.feature_order[1]+1):#3
                        tmp_l = tmp_k[tmp_k[:,3]==l]
                        
                        for m in range(self.feature_order[1]+1):#4
                            tmp_m = tmp_l[tmp_l[:,4]==m]
                            
                            ijkl_indx = np.argsort(tmp_m[:,6])#6
        #                     print('ijk_indx' + str(len(ijk_indx)))
                            if i == self.feature_order[1] and j ==1 and k ==0: # flag に書き換え
                                ijkl_product_mat=tmp_m[ijkl_indx]
        #                             print(ijkl_product_mat)
                            else:
                                ijkl_product_mat = np.concatenate([ijkl_product_mat, tmp_m[ijkl_indx]], axis=0)
#         print(ijk_product_mat)
        product_mat = ijkl_product_mat

        return product_mat