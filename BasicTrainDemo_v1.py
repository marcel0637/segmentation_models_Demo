from segmentation_models import Unet
from PIL import Image
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans

def load_png_files(image_path,start,end): # 加载图片成numpy的list
    im_array = []
    for i in range(start,end+1):
        im = Image.open(os.path.join(image_path,str(i)+'.png'))
        tmp_array = np.array(im) #图片转numpy数组
        tmp_array = tmp_array[np.newaxis,:,:] #numpy数组添加一维,为了把二维图片转成三维图片集
        if len(im_array) == 0:
            im_array = tmp_array
        else:
            im_array = np.concatenate((im_array,tmp_array),axis=0) #将新的图片加入到之前的图片集
    return im_array

def main():

    image_path = "image" #加载训练图片
    im_start = 0 #确定编号
    im_end = 29
    im_array = load_png_files(image_path,im_start,im_end)
    im_array = im_array[:,:,:,np.newaxis] # 需要加代表图片的通道数,这里是黑白图片所以是1,因此直接加一维
    print("train_image shape : " + im_array.shape)

    label_path = "label" #加载训练图片对应的标签图片
    la_start = 0
    la_end = 29
    la_array = load_png_files(label_path,la_start,la_end)
    la_array = la_array[:,:,:,np.newaxis]
    print("train_label shape : " + la_array.shape)

    test_path = "test" #加载测试集的图片
    te_start = 0
    te_end = 4
    te_array = load_png_files(test_path,te_start,te_end)
    te_array = te_array[:,:,:,np.newaxis]
    print("test_image shape : " + te_array.shape)

    model = Unet('resnet34', input_shape = (512, 512, 1), encoder_weights = None) #1代表通道数
    model.compile('Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(
        x = im_array,
        y = la_array,
        batch_size = 10,
        epochs = 8,
        validation_split = 0.2, #取训练集中的0.2作为验证集
        shuffle = True
    )
    model.save("model_v1.h5") #保存模型 

    print("Saved model to disk")
    print("done!!!!!")
   
if __name__ == '__main__':
    main()