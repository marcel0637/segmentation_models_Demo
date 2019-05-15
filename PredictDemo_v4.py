from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt


#给定路径,起点和终点,进行图像增强,每次迭代返回一个batch_size的测试图片集和标签集 -> 这里的起点和终点对应下标都是包含在内的!
def test_image_generator(image_path,st,ed,batch_size): 
    nowinx = st #设定初始图片
    while True:
        im_array = []
        for i in range(batch_size):
            im = Image.open(os.path.join(image_path,str(nowinx)+'.png'))
            tmp_im_array = np.array(im) #图片转numpy数组
            tmp_im_array = tmp_im_array / 255 #对数据进行归一化
            tmp_im_array = tmp_im_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集

            if len(im_array) == 0:
                im_array = tmp_im_array
            else:
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) #将新的图片加入到之前的图片集
            
            nowinx = st if nowinx==ed else nowinx+1 #如果遍历到了超出最后一个时,就返回到第一个

        im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.

        yield(im_array) #按批次返回数据

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile): #取出预测结果中的每一个
        im_np = item[:,:,0]
        im_np[im_np > 0.5] = 1
        im_np[im_np <= 0.5] = 0
        io.imsave(os.path.join(save_path,"%d_predict_v4.png"%i),im_np)

model = load_model("model_v4.h5") #加载模型

test_gen = test_image_generator("test",0,4,1) # 获取一个测试数据生成器,一次生成一个
res = model.predict_generator(test_gen,5) # 进行结果预测,5代表预测的个数
saveResult("res",res)