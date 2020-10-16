import keras
from keras.models import *
from keras.layers import *
from AdaIN import AdaInstanceNormalization
import cv2
import numpy as np



def loadModel(name): #Load a Model
        
        file = open("./gan_test/Model/"+"gen_c13"+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("./gan_test/Model/"+name+".h5")
        
        return mod
    
    
        

def load(): #Load JSON and Weights from /Models/
    
    gen = loadModel("gen_c13_464")

    return gen


model = load()

def noise():
    return np.random.normal(0.0, 1.0, size = [1, 512])

noisew = noise()
noisew2 = noise()

img = cv2.imread('./gan_test/img/2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)

img2 = cv2.imread('./gan_test/img/3.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
noiseimage = img.reshape(1,256,256,-1)/255
noiseimage2 = img2.reshape(1,256,256,-1)/255

print(noisew.shape)
print(noiseimage.shape)

# model.summary()
# p_dog = model.predict([noisew, noiseimage, np.ones((1,1))])
# p2_dog = model.predict([noisew2, noiseimage, np.ones((1,1))])

import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import linspace
def plot_vector(examples,n,k):
    for i in range(n):
        pyplot.subplot(1,n,i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i].reshape(256,256,3))
    # plt.tight_layout()
    
    plt.savefig("test_{}.png".format(k),dpi=1028)
    # plt.show()
def dog_vector(img,img2,p,p2,model):

    num = 10
    examples = []
    examples2 = []

    ratios = linspace(0,1,30)
    for i in ratios:
        
        noise = (1 - i) * p + i * p2
        examples.append(model.predict([noise, img, np.ones((1,1))]))
        examples2.append(model.predict([noise, img2, np.ones((1,1))]))


    plot_vector(examples,30,1)
    plot_vector(examples2,30,2)

    
    
dog_vector(noiseimage, noiseimage2 , noisew, noisew2, model=model)
# plt.imshow(p_dog.reshape((256,256,3)))
# plt.show()