#coding=utf-8
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt

#给定路径,起点和终点,进行图像增强,每次迭代返回一个batch_size的训练图片集和标签集 -> 这里的起点和终点对应下标都是包含在内的!
def train_image_generator(image_path,label_path,aug = None): 
    im_array = []
    lb_array = []
    im = Image.open(os.path.join(image_path,str(0)+'.png'))
    tmp_im_array = np.array(im) #图片转numpy数组
    tmp_im_array = tmp_im_array / 255 #对数组进行压缩
    tmp_im_array = tmp_im_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集
    
    lb = Image.open(os.path.join(label_path,str(0)+'.png'))
    tmp_lb_array = np.array(lb) #图片转numpy数组
    tmp_lb_array = tmp_lb_array / 255
    # tmp_lb_array[tmp_lb_array > 0.6] = 1
    # tmp_lb_array[tmp_lb_array <= 0.6] = 0 #对数组进行压缩并且变成01分布的数组
    tmp_lb_array = tmp_lb_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集
    
    if len(im_array) == 0:
        im_array = tmp_im_array
        lb_array = tmp_lb_array
    else:
        im_array = np.concatenate((im_array,tmp_im_array),axis=0) #将新的图片加入到之前的图片集
        lb_array = np.concatenate((lb_array,tmp_lb_array),axis=0) #将新的图片加入到之前的图片集

    im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.
    lb_array = lb_array[:,:,:,np.newaxis]

    if aug is not None : #如果传入了数据增强的生成器,就进行数据增强
        print("do it!")
        new_array = im_array
        new_array = np.concatenate((new_array,lb_array),axis=3)
        new_array = next(aug.flow(new_array,batch_size = 1))
        im_array = new_array[:,:,:,0]
        lb_array = new_array[:,:,:,1]
        im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.
        lb_array = lb_array[:,:,:,np.newaxis]
    print(im_array.shape)
    im_array = im_array[0,:]
    im_array = im_array[:,:,0]
    lb_array = lb_array[0,:]
    lb_array = lb_array[:,:,0]
    io.imsave("t1.png",im_array)
    io.imsave("t2.png",lb_array)

aug = ImageDataGenerator( #定义一个数据增强生成器
    rotation_range = 0.05, # 定义旋转范围
    zoom_range = 0.05, # 按比例随机缩放图像尺寸
	width_shift_range = 0.05, # 图片水平偏移幅度
    height_shift_range = 0.05, # 图片竖直偏移幅度
    shear_range = 0.05, # 水平或垂直投影变换
	horizontal_flip = True, # 水平翻转图像
    fill_mode = "reflect" # 填充像素,出现在旋转或平移之后
)

train_gen = train_image_generator("image","label",aug) # 获取一个训练数据生成器

print("done!!!!!")
