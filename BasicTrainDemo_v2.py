#coding=utf-8
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.optimizers import *
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt

#给定路径,起点和终点,进行图像增强,每次迭代返回一个batch_size的训练图片集和标签集 -> 这里的起点和终点对应下标都是包含在内的!
def train_image_generator(image_path,label_path,st,ed,batch_size,aug = None): 
    nowinx = st #设定初始图片
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            im = Image.open(os.path.join(image_path,str(nowinx)+'.png'))
            tmp_im_array = np.array(im) #图片转numpy数组
            tmp_im_array = tmp_im_array / 255 #对数组进行压缩
            tmp_im_array = tmp_im_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集
            
            lb = Image.open(os.path.join(label_path,str(nowinx)+'.png'))
            tmp_lb_array = np.array(lb) #图片转numpy数组
            tmp_lb_array = tmp_lb_array / 255
            tmp_lb_array[tmp_lb_array > 0.6] = 1
            tmp_lb_array[tmp_lb_array <= 0.6] = 0 #对数组进行压缩并且变成01分布的数组
            tmp_lb_array = tmp_lb_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集
            
            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array = tmp_lb_array
            else:
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) #将新的图片加入到之前的图片集
                lb_array = np.concatenate((lb_array,tmp_lb_array),axis=0) #将新的图片加入到之前的图片集
            
            nowinx = st if nowinx==ed else nowinx+1 #如果遍历到了超出最后一个时,就返回到第一个

        im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.
        lb_array = lb_array[:,:,:,np.newaxis]

        if aug is not None : #如果传入了数据增强的生成器,就进行数据增强
            (im_array,lb_array) = next(aug.flow(im_array,lb_array,batch_size = batch_size))

        yield(im_array,lb_array) #按批次返回数据

aug = ImageDataGenerator( #定义一个数据增强生成器
    rotation_range = 0.2, # 定义旋转范围
    zoom_range = 0.05, # 按比例随机缩放图像尺寸
	width_shift_range = 0.05, # 图片水平偏移幅度
    height_shift_range = 0.05, # 图片竖直偏移幅度
    shear_range = 0.05, # 水平或垂直投影变换
	horizontal_flip = True, # 水平翻转图像
    fill_mode = "nearest" # 填充像素,出现在旋转或平移之后
)

train_gen = train_image_generator("image","label",0,29,2,aug) # 获取一个训练数据生成器
#validate_gen = train_image_generator("image","label",21,29,3,None) # 获取一个验证数据生成器

#定义并编译一个模型
model = Unet('resnet34', input_shape = (512, 512, 1), encoder_weights = None) #1代表通道数
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

#进行模型的训练,并用his来记录训练过程中的参数变化,方便最后生成图像
his = model.fit_generator( 
    generator = train_gen, #训练集生成器
    steps_per_epoch = 100, #训练集每次epoch的数量
    # validation_data = validate_gen, #验证集生成器
    # validation_steps = 3, #验证集每次epoch的数量
    epochs = 1 #进行epoch的次数
)

model.save("model_v2_060.h5") #保存模型 

print("Saved model to disk")


# 生成训练参数图片
# N = 51
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), his.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), his.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), his.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), his.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")

print("done!!!!!")
