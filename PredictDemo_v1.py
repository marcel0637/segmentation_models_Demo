from segmentation_models import Unet
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans

def load_png_files(image_path,start,end): # 加载图片成numpy的list
    im_array = []
    for i in range(start,end+1):
        im = Image.open(os.path.join(image_path,str(i)+'.png'))
        tmp_array = np.array(im)
        tmp_array = tmp_array[np.newaxis,:,:]
        if len(im_array) == 0:
            im_array = tmp_array
        else:
            im_array = np.concatenate((im_array,tmp_array),axis=0)
    return im_array

def evaluate_res(model,res):
    rs_path = "res"
    rs_start = 0
    rs_end = 4
    rs_array = load_png_files(rs_path,rs_start,rs_end)
    rs_array = rs_array[:,:,:,np.newaxis]
    for a in range(0,res.shape[0]):
        for b in range(0,res.shape[1]):
            for c in range(0,res.shape[2]):
                for d in range(0,res.shape[3]):
                    res[a][b][c][d]*=255.0
    ress_array = res.astype(int)
    print(ress_array.shape)
    print(model.evaluate(ress_array,rs_array,5))


test_path = "test" #加载测试集图片的标签图片
te_start = 0
te_end = 4
te_array = load_png_files(test_path,te_start,te_end)
te_array = te_array[:,:,:,np.newaxis]

model = load_model("model_v1.h5") #加载模型

res=model.predict(te_array,5) #进行预测,5代表size?

print(res.shape)

#evaluate_res(model,res) #计算准确度

# for i in range(res.shape[0]):
#     im_np = np.squeeze(np.squeeze(res[i,:,:])) # 去掉第一维
#     for a in range(0,im_np.shape[0]):
#         for b in range(0,im_np.shape[1]):
#             im_np[a][b] *= 255.0 #把概率矩阵变成灰度矩阵
#     im_npp = im_np.astype(int)
#     print("图片{name} :".format(name=i))
#     print(im_npp)
#     print(im_npp.shape)
#     img = Image.fromarray(im_npp)
#     img = img.convert("RGB") # 改变图像模式,黑白为L,彩色为RGB
#     img.save("res/"+str(i)+"_predict_v1.png")

for i in range(res.shape[0]):
    im_np = np.squeeze(np.squeeze(res[i,:,:])) # 去掉第一维
    for a in range(0,im_np.shape[0]):
        for b in range(0,im_np.shape[1]):
            im_np[a][b] *= 255
    im_npp = im_np.astype(int)
    print("图片{name} :".format(name=i))
    print(im_npp)
    print(im_npp.shape)
    img = Image.fromarray(im_npp)
    img = img.convert("L") # 改变图像模式,黑白为L,彩色为RGB
    img.save("res/"+str(i)+"_predict_v1.png")