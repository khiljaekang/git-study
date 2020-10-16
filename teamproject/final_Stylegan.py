
from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial
from random import random
import h5py
import os
# from functools import partial

#Config Stuff
im_size = 256           # image_size
latent_size = 512      # noise size
BATCH_SIZE = 8        
directory1 = "70400"  # image 폴더
n_images1 = 7000         # data 수
suff1 = 'png'            # image 확장자명

directory2 = "data/hdf5/DATA.hdf5"  # image 폴더
n_images2 = 7000        # data 수
suff2 = 'jpg'            # image 확장자명

foldername = 'jun2'

#Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])

#Get random samples from an array
def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

#Import Images Function 
def import_images(loc, flip = True, suffix = 'png'):
    
    out = []
    cont = True
    i = 1
    print("Importing Images...")
    
    while(cont):
        try:
            temp = Image.open("data/"+loc+"/im ("+str(i)+")."+suffix+"").convert('RGB')
            temp = temp.resize((im_size, im_size), Image.BICUBIC)
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))
            
            i = i + 1
        except:
            cont = False
        
    print(str(i-1) + " images imported.")
            
    return np.array(out)

def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

#This is the REAL data generator, which can take images from disk and temporarily use them in your program.
#Probably could/should get optimized at some point
class dataGenerator(object):
    
    def __init__(self, loc, n, flip = True, suffix = 'png'):
        self.loc = "D:/"+loc       # data 경로
        self.flip = flip
        self.suffix = suffix            # 확장자
        self.n = n                      # 데이터 양
    
    def get_batch(self, amount):
        
        idx = np.random.randint(0, self.n - 1, amount) # + 1    # 랜덤 인덱스 ( 0, 데이터양, 배치싸이즈(특정 개수만 랜덤으로 뽑음))
        out = []                                                # 배치 이미지 저장
        
        for i in idx:
            # print(i)
            if self.loc == "D:/"+directory2:                  #강아지 데이터 DATA (파일위치 )
                file = h5py.File(self.loc, 'r') 
                temp = file['256'][i]
                temp1 = np.array(temp)
            else:                                             #폴더에서 이미지 불러옴  FFHQ
                temp = Image.open(self.loc+"/{0:05d}.".format(i)+self.suffix+"").convert('RGB') #0:05d 0번쨰부터 5번쨰 자리까지 
            # temp = Image.open(self.loc+"/00000."+self.suffix+"").convert('RGB')
                temp1 = temp.resize((256, 256))
                temp1 = np.array(temp1.convert('RGB'), dtype='float32') / 255
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)                           # axis = 1 로 뒤집기
                
            out.append(temp1)
            
        
        return np.array(out)

#Imports for layers and models
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D, MaxPooling2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from AdaIN import AdaInstanceNormalization



#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight, sample_weight = None):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    print(gradients)
    gradients_sqr = K.square(gradients)
    print(gradients_sqr)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))  
    print(gradient_penalty)                        
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)

#---------------------------------------------------------------------------------------------------------------

#Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block(inp, style, noise, fil, u = True):
    
    b = Dense(fil)(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    
    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = Activation('linear')(inp)
    
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)
    
    b = Dense(fil)(style)                       
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)
    
    return out


#Convolution, Activation, Pooling, Convolution, Activation
def d_block(inp, fil, p = True):    # inp = input_layer / fil = filter / p = pooling 
    
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
    out = LeakyReLU(0.01)(route2)
    
    return out

#---------------------------------------------------------------------------------------------------------------

#This object holds the models
class GAN(object):
    
    def __init__(self, lr = 0.0001):
        
        #Models
        self.D = None
        self.G = None
        
        self.DM = None
        self.AM = None
        
        #Config
        self.LR = lr
        self.steps = 1
        
        #Init Models
        self.discriminator()
        self.generator()
        
    def discriminator(self):
        
        if self.D:
            return self.D
        
        inp = Input(shape = [im_size, im_size, 3])      # (256, 256, 3)
        
        # Size
        x = d_block(inp, 16) # Size / 2 -> filter = 16, AveragePooling 
        x = d_block(x, 32) # Size / 4
        x = d_block(x, 64) # Size / 8
        
        if (im_size > 32):
            x = d_block(x, 128) #Size / 16
        
        if (im_size > 64):
            x = d_block(x, 192) #Size / 32
        
        if (im_size > 128):
            x = d_block(x, 256) #Size / 64
        
        if (im_size > 256):
            x = d_block(x, 384) #Size / 128
            
        if (im_size > 512):
            x = d_block(x, 512) #Size / 256
            
            
        x = Flatten()(x)    # 2차원 out
        
        x = Dense(128)(x)
        x = Activation('relu')(x)   # ? 왜 sigmoid 아님
        
        x = Dropout(0.6)(x)
        x = Dense(1)(x)             # Real / Fake
        
        self.D = Model(inputs = inp, outputs = x)   # 함수형 모델
        
        return self.D
    
    def generator(self):
        
        if self.G:
            return self.G
        
        #Style FC, I only used 2 fully connected layers instead of 8 for faster training
        

        inp_s = Input(shape = [latent_size])
        sty = Dense(512, kernel_initializer = 'he_normal')(inp_s)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)
        

        #Get the noise image and crop for each size
        inp_n = Input(shape = [im_size, im_size, 3])      
        noi_n = Conv2D(1, 3, padding='same')(inp_n)          # noiseImage
        noi = [Activation('linear')(noi_n)]
        curr_size = im_size
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size/2))(noi[-1]))
        
        #Here do the actual generation stuff                        # 실제 생성 작업
        inp = Input(shape = [1])
        x = Dense(4 * 4 * 512, kernel_initializer = 'he_normal')(inp)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, sty, noi[-1], 512, u=False)
        
        if(im_size >= 1024):
            x = g_block(x, sty, noi[7], 512) # Size / 64
        if(im_size >= 512):
            x = g_block(x, sty, noi[6], 384) # Size / 64
        if(im_size >= 256):
            x = g_block(x, sty, noi[5], 256) # Size / 32 
        if(im_size >= 128):
            x = g_block(x, sty, noi[4], 192) # Size / 16
        if(im_size >= 64):
            x = g_block(x, sty, noi[3], 128) # Size / 8
            
        x = g_block(x, sty, noi[2], 64) # Size / 4
        x = g_block(x, sty, noi[1], 32) # Size / 2
        x = g_block(x, sty, noi[0], 16) # Size
        
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
        
        self.G = Model(inputs = [inp_s, inp_n, inp], outputs = x)
        
        return self.G
    
    def AdModel(self):
        
        #D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False
        
        #G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True
        
        #This model is simple sequential one with inputs and outputs
        gi = Input(shape = [latent_size])
        gi2 = Input(shape = [im_size, im_size, 3])
        gi3 = Input(shape = [1])
        
        gf = self.G([gi, gi2, gi3])
        df = self.D(gf)
        
        self.AM = Model(inputs = [gi, gi2, gi3], outputs = df)
            
        self.AM.compile(optimizer = Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')
        
        return self.AM
    
    def DisModel(self):
        
        #D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True
        
        #G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False
        
        # Real Pipeline
        ri = Input(shape = [im_size, im_size, 3])   # real_image
        dr = self.D(ri)
        
        # Fake Pipeline
        gi = Input(shape = [latent_size])           # noise
        gi2 = Input(shape = [im_size, im_size, 3])  # noiseImage
        gi3 = Input(shape = [1])                    # ones

        gf = self.G([gi, gi2, gi3])
        df = self.D(gf)
        
        # Samples for gradient penalty
        # For r1 use real samples (ri)
        # For r2 use fake samples (gf)
        da = self.D(ri)
        
        # Model With Inputs and Outputs
        self.DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, da])
        
        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 5)
        
        #Compile With Corresponding Loss Functions
        self.DM.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])
        
        return self.DM
        
        

class WGAN(object):
    
    def __init__(self, steps = -1, lr = 0.0001, silent = True):
        
        self.GAN = GAN(lr = lr)
        self.DisModel = self.GAN.DisModel()         # 판별자
        self.AdModel = self.GAN.AdModel()           # AdaIn
        self.generator = self.GAN.generator()       # 생성자
        
        if steps >= 0:
            self.GAN.steps = steps                  
        
        self.lastblip = time.clock()                # 현재 시간
        
        self.noise_level = 0            
        
        #self.ImagesA = import_images(directory, True)
        self.im = dataGenerator(directory2, n_images2, suffix = suff2, flip = True)    # 어디서 어떻게 가져올지 정함
        self.noiseperson = dataGenerator(directory1, n_images1, suffix = suff1, flip = True)
        #(self.im, _), (_, _) = cifar10.load_data()
        #self.im = np.float32(self.im) / 255
        
        self.silent = silent

        #Train Generator to be in the middle, not all the way at real. Apparently works better??
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)          # label : Real
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)        #       : Fake
        self.nones = -self.ones
        
        self.enoise = noise(8)             
        self.enoiseImage = self.noiseperson.get_batch(8)    # (n, 256, 256, 1)
    
    #---------------------------------------------------------------------------------------------------------------

    def train(self):
        
        #Train Alternating
        a = self.train_dis()                
        self.b = self.train_gen()
        
        #Print info
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(self.b))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()

            self.evalTrunc(self.GAN.steps)                             # image save
            print('--------------------'+str(foldername)+'-------------------')
            
            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 1000))    # floor :내림 / round : 반올림
            if self.GAN.steps % 1000 == 0:
                self.evaluate(floor(self.GAN.steps / 1000))
            
        
        self.GAN.steps = self.GAN.steps + 1
          
    

    def Adain_image(self, real, noisew):
        # real image 
        # 사람 이미지 (noise)
        # keras backend를 사용하기 위해서 numpy -> tensor로 변환
        x = tf.convert_to_tensor(real, np.float32)
        y = tf.convert_to_tensor(noisew, np.float32)

        axis = -1
        epsilon=1e-3

        input_shape = K.int_shape(x)
        # print(input_shape)               # (400, 750, 3)

        reduction_axes = list(range(0, len(input_shape)))
        # print(reduction_axes)              # [0, 1, 2]

        if axis is not None:
            del reduction_axes[axis]

        del reduction_axes[0]

        # print(reduction_axes)

        gamma = K.std(y, reduction_axes, keepdims=True) + epsilon   # S(y)
        beta = K.mean(y, reduction_axes, keepdims=True)             # mean(y)
        # print(beta.shape)                   # (1, 1, 1, 3)
        # print(gamma.shape)                  # (1, 1, 1, 3)

        # Adain
        mean = K.mean(x, reduction_axes, keepdims=True)             # mean(x)
        stddev = K.std(x, reduction_axes, keepdims=True) + epsilon  # S(x)
        normed = (x - mean) / stddev

        # print(mean.shape)                   # (1, 1, 1, 3)
        # print(stddev.shape)                 # (1, 1, 1, 3)
        # print(normed.shape)                 # (1, 400, 750, 3)

        # result
        image = normed * gamma + beta
        return np.array(image)

    def train_dis(self):
        
        real = self.im.get_batch(BATCH_SIZE)
        noisew = noise(BATCH_SIZE)
        noiseimage = self.noiseperson.get_batch(BATCH_SIZE)

        adain_image = self.Adain_image(real, noiseimage)
        #Get Data
        train_data = [real , noisew, noiseimage, self.ones]
        train_data2 = [adain_image, noisew, noiseimage, self.ones] # adain 섞은 real image로 학습 데이터 생성
    
        
        #Train
        d_loss_adain = self.DisModel.train_on_batch(train_data2, [self.ones, self.nones, self.ones])
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])
        print(d_loss)
        return [x + y for x, y in zip([0.3 * i for i in d_loss], [0.7 * i for i in d_loss_adain])]
       
    def train_gen(self):
        
        #Train
        g_loss = self.AdModel.train_on_batch([noise(BATCH_SIZE), self.noiseperson.get_batch(BATCH_SIZE), self.ones], self.zeros)
        
        return g_loss

    #---------------------------------------------------------------------------------------------------------------
    
    def evaluate(self, num = 0, trunc = 2.0): #8x4 images, bottom row is constant
        
        n = noise(32)
        n2 = self.noiseperson.get_batch(32)
        
        im2 = self.generator.predict([n, n2, np.ones([32, 1])])
        im3 = self.generator.predict([self.enoise, self.enoiseImage, np.ones([8, 1])])
        
        r12 = np.concatenate(im2[:8], axis = 1)
        r22 = np.concatenate(im2[8:16], axis = 1)
        r32 = np.concatenate(im2[16:24], axis = 1)
        r43 = np.concatenate(im3[:8], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))               # .fromarray() : numpy -> image
        
        x.save("D:/GAN/Results/"+str(foldername)+"i"+str(num)+".jpg")
        
    def evaluate2(self, s1, s2, n1, n2, num = 0, weight = 0.5):
        
        s = normalize((s2 * weight) + (s1 * (1 - weight)))
        n = (n2 * weight) + (n1 * (1 - weight))
        
        im2 = self.generator.predict([s, n, np.ones([32, 1])])
        
        r12 = np.concatenate(im2[:8], axis = 1)
        r22 = np.concatenate(im2[8:16], axis = 1)
        r32 = np.concatenate(im2[16:24], axis = 1)
        r43 = np.concatenate(im2[24:], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save("D:/GAN/Results/"+str(foldername)+"i"+str(num)+".jpg")
        
    def evalTrunc(self, num = 0, trunc = 1.8):
        
        n = np.clip(noise(16), -trunc, trunc)       # np.clip(배열, min, max) : 최솟값과 최댓값 기준으로 일괄 처리 
        n2 = self.noiseperson.get_batch(16)
        
        im2 = self.generator.predict([n, n2, np.ones([16, 1])])
        
        r12 = np.concatenate(im2[:4], axis = 1)     # 4 x 4 이미지
        r22 = np.concatenate(im2[4:8], axis = 1)
        r32 = np.concatenate(im2[8:12], axis = 1)
        r43 = np.concatenate(im2[12:], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save("D:/GAN/Results/"+str(foldername)+"/t"+str(num)+'_loss%.6f.jpg'%(self.b))         # image 저장 -> 경로

        
    
    def saveModel(self, model, name, num): #Save a Model

        json = model.to_json()
        with open("D:/GAN/Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("D:/GAN/Models/"+name+"_"+str(num)+".h5")
        
    def loadModel(self, name, num): #Load a Model
        
        file = open("D:/GAN/Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("D:/GAN/Models/"+name+"_"+str(num)+".h5")
        
        return mod
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen_c13", num)
        self.saveModel(self.GAN.D, "dis_c13", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.G = self.loadModel("gen_c13", num)
        self.GAN.D = self.loadModel("dis_c13", num)
        
        self.GAN.steps = steps1
        
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        

import keras.backend.tensorflow_backend as K
        
if __name__ == "__main__":
    model = WGAN(lr = 0.0002, silent = False)
    # model.load(101)
    cnt = 0
    while True:
        with K.tf.device('/gpu:1'):
            model.train()

        '''
        # for i in range(10):
            if  cnt % 20 ==0:
                model.evalTrunc(cnt)

            cnt+=1
            
            
            if cnt == 50000:
                break
        '''