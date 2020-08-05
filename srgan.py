import tensorflow as tf
from keras import Input
from keras.applications.vgg19 import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
import glob
import time
import os
import cv2
import base64
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from imageio import imread
from skimage.transform import resize as imresize
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint
from PIL import Image
from sklearn.model_selection import train_test_split

class Srgan :
    
    def __init__(self,data_dir,epochs,path1,path2,path3):
        
        self.data_dir=data_dir
        self.epochs=epochs
        self.path1=path1
        self.path2=path2
        self.path3=path3
        
    
    def build_generator(self):
        
        def residual_block(x):
            filters = [64, 64]
            #filters = [128, 128]
            kernel_size = 3
            strides = 1
            padding = "same"
            momentum = 0.8
            activation = "relu"
            res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
            res = Activation(activation=activation)(res)
            res = BatchNormalization(momentum=momentum)(res)
            res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
            res = BatchNormalization(momentum=momentum)(res)
            res = Add()([res, x])
            return res
    
    
        residual_blocks = 16
        momentum = 0.8
        input_shape = (64, 64, 3)
        input_layer = Input(shape=input_shape)
        gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)
        # add 16 residual blocks
        res = residual_block(gen1)
        for i in range(residual_blocks - 1):
            res = residual_block(res)
            # post-residual block: convolutional layer and batch-norm layer after residual blocks
        gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
        gen2 = BatchNormalization(momentum=momentum)(gen2)
        # take the sum of pre-residual block(gen1) and post-residual block(gen2)
        gen3 = Add()([gen2, gen1])
        # UpSampling: learning to increase dimensionality
        gen4 = UpSampling2D(size=2)(gen3)
        gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
        gen4 = Activation('relu')(gen4)
        # UpSampling: learning to increase dimensionality
        gen5 = UpSampling2D(size=2)(gen4)
        gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
        gen5 = Activation('relu')(gen5)
        # convolution layer at the output
        gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
        output = Activation('tanh')(gen6)
        # model 
        model = Model(inputs=[input_layer], outputs=[output], name='generator')
        return model
    
    def build_discriminator(self):
        #define hyper-parameters
        leakyrelu_alpha = 0.2
        momentum = 0.8
        # dimentions correspond to HR - High Resolution
        input_shape = (256, 256, 3)
        # input layer for discriminator
        input_layer = Input(shape=input_shape)
        # 8 convolutional layers with batch normalization  
        dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
        dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

        dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
        dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
        dis2 = BatchNormalization(momentum=momentum)(dis2)

        dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
        dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
        dis3 = BatchNormalization(momentum=momentum)(dis3)

        dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
        dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
        dis4 = BatchNormalization(momentum=0.8)(dis4)

        dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
        dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
        dis5 = BatchNormalization(momentum=momentum)(dis5)

        dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
        dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
        dis6 = BatchNormalization(momentum=momentum)(dis6)

        dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
        dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
        dis7 = BatchNormalization(momentum=momentum)(dis7)

        dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
        dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
        dis8 = BatchNormalization(momentum=momentum)(dis8)
    
        # fully-connected layer 
        dis9 = Dense(units=1024)(dis8)
        dis9 = LeakyReLU(alpha=0.2)(dis9)
    
        # last fully-connected layer - for classification 
        output = Dense(units=1, activation='sigmoid')(dis9)
        model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
        return model
    
    def build_vgg(self):
        # dimensionality correspond to HR - High Resolution
        input_shape = (256, 256, 3)
        # upload VGG19 network pre-trained on 'Imagenet'
        vgg = VGG19(weights="imagenet")
        # take input from the 9-th layer
        vgg.outputs = [vgg.layers[9].output]
        # input layer
        input_layer = Input(shape=input_shape)
        # extract features 
        features = vgg(input_layer)
        # model
        model = Model(inputs=[input_layer], outputs=[features])
        return model
    
    
    def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
        all_images = glob.glob(data_dir+'*.jpg')
        images_batch = np.random.choice(all_images, size=batch_size)
        low_resolution_images = []
        high_resolution_images = []
        for img in images_batch:
            img1 = imread(img, as_gray=False, pilmode='RGB')
            img1 = img1.astype(np.float32)
            img1_high_resolution = imresize(img1, high_resolution_shape)
            img1_low_resolution = imresize(img1, low_resolution_shape)
            if np.random.random() < 0.5:
                img1_high_resolution = np.fliplr(img1_high_resolution)
                img1_low_resolution = np.fliplr(img1_low_resolution)
            high_resolution_images.append(img1_high_resolution)
            low_resolution_images.append(img1_low_resolution)
        return np.array(high_resolution_images), np.array(low_resolution_images)
    
    def build_adversarial_model(generator, discriminator, vgg):
        low_resolution_shape = (64, 64, 3)
        high_resolution_shape = (256, 256, 3)
        input_high_resolution = Input(shape=high_resolution_shape)
        input_low_resolution = Input(shape=low_resolution_shape)
        generated_high_resolution_images =generator(input_low_resolution)
        #print(generated_high_resolution_images)
        features = vgg(generated_high_resolution_images)
        discriminator.trainable = False
        discriminator.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
        probs = discriminator(generated_high_resolution_images)
        adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
        adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer="adam")
        return adversarial_model

class Srgan1(Srgan) :

    def save_images(original_image,path):
        #plt.imshow(original_image)
        plt.savefig(path)
    
    def save_images1(low_resolution_images,path):
        #plt.imshow(low_resolution_images)
        plt.savefig(path)
    
    def save_images2(generated_img,path):
         #plt.imshow(generated_img)
            plt.savefig(path)
    
    def srgan(self):
        
        vgg =Srgan.build_vgg(self)
        vgg.trainable = False
        vgg.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
        
        discriminator =Srgan.build_discriminator(self)
        discriminator.trainable = True
        discriminator.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
        
        generator = Srgan.build_generator(self)
        adversarial_model =Srgan.build_adversarial_model(generator, discriminator, vgg)

        for epoch in range(self.epochs):
            
            d_history = []
            g_history = []
            print("Epoch:{}".format(epoch))
            low_resolution_shape = (64, 64, 3)
            high_resolution_shape = (256, 256, 3)
            high_resolution_images, low_resolution_images =Srgan.sample_images(data_dir=self.data_dir,batch_size=1,low_resolution_shape=low_resolution_shape,high_resolution_shape=high_resolution_shape)
            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.
            generated_high_resolution_images = generator.predict(low_resolution_images)
            real_labels = np.ones((1, 16, 16, 1))
            fake_labels = np.zeros((1, 16, 16, 1))
            d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
            d_loss_real =  np.mean(d_loss_real)
            d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
            d_loss_fake =  np.mean(d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_history.append(d_loss)
            print("D_loss:", d_loss)
            high_resolution_images, low_resolution_images =Srgan.sample_images(data_dir=self.data_dir, batch_size=1,low_resolution_shape=low_resolution_shape,high_resolution_shape=high_resolution_shape)
            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.
            image_features = vgg.predict(high_resolution_images)
            g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],[real_labels, image_features])
            g_history.append( 0.5 * (g_loss[1]) )
            print( "G_loss:", 0.5 * (g_loss[1]) )
            if epoch==99:
                high_resolution_images, low_resolution_images =srgan.sample_images(data_dir=self.data_dir, batch_size=1,low_resolution_shape=low_resolution_shape,high_resolution_shape=high_resolution_shape)
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.
                generated_images = generator.predict(low_resolution_images)
                for index, img in enumerate(generated_images):
                    save_images(high_resolution_images[index],path=path1+"/img_{}_{}".format(epoch,index))
                    save_images1(low_resolution_images[index],path=path2+"/img_{}_{}".format(epoch, index))
                    save_images2(generated_images,path=path3+"/img_{}_{}".format(epoch, index))
            
if __name__ == "__main__":
    
    data_dir=input('Enter the Input image directory:')
    
    epochs=int(input('Enter the number of epochs you want to run:'))
    
    path1= input('Enter the path to store original images:')
    
    path2=input('Enter the path to store low resolution images:')
    
    path3=input('Enter the path to store generated  images:')
    
    a=Srgan1(data_dir,epochs,path1,path2,path3)
    
    b=a.srgan()
