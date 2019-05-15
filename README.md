### 前言

本文基于下面链接的项目, 实践一次基于Unet模型的图像分割. 

个人在实现时遇到很多问题, 许多问题在网上并没有找到解答, 所以写下本文, 记录了在实现过程中遇到的问题和大部分代码的注解, 主要从代码逻辑入手, 分析整个实践过程. 

我的实现代码放在文章最后, 供大家参考

[参考链接](https://blog.csdn.net/u012931582/article/details/70215756)



### 效果展示

![](<https://raw.githubusercontent.com/marcel0637/segmentation_models_Demo/master/markdownsrc/img.png>)

![](<https://github.com/marcel0637/segmentation_models_Demo/blob/master/markdownsrc/mask.png?raw=true>)



### 一些声明

- 本文实现的代码仅考虑灰度图, 即单通道图

- 本文中涉及到的数据增强, Unet模型部分, 均为直接调用API, 只讲解具体怎么使用, 涉及具体原理部分需要自己在网上查询

- 我的实现写成了多个版本, 这里讲解第一个和第四个版本

- 代码运行在Google Colab上, 这里放一下网上别人的使用教程 ( 使用Colab需要科学上网, 这样比较稳 )

  [Colab使用参考链接](https://www.jianshu.com/p/000d2a9d36a0)

### 整体概述

整个项目需要实现的是对图像的分割, 为二分类问题, 因此我们主要的代码逻辑如下 : 

1. 读入数据集和标签, 进行数据增强(也可不进行数据增强)
2. 预编译一个loss函数为binary_crossentropy的Unet模型
3. 使用数据和标签对Unet进行训练, 将训练好的模型进行保存
4. 加载模型进行预测和图片生成



### 读入数据

首先明确一下, 我们的图像和标签都是 512 x 512 的图片. 

我们首先将图片转换成numpy数组并将多张图片合到一起, 此时变成了三维的数组 : n x 512 x 512 

```python
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
```

由于我们将数据用于训练时还需要一维的通道, 所以使用以下代码增加一维

```python
im_array = im_array[:,:,:,np.newaxis]
```

注意 : 此时用于训练的数据均是 0~255的 uint 型的数据



### 预编译模型

```
model = Unet('resnet34', input_shape = (512, 512, 1), encoder_weights = None)#1代表通道数  model.compile('Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

这里创建了一个Unet模型并进行了预编译, 具体参数个人感觉比较好理解, 有不懂的话这里提供中文文档供查阅

[Keras-model中文文档参考链接](https://keras-cn.readthedocs.io/en/latest/legacy/models/model/#_1)



### 模型训练和保存

```python
model.fit(
    x = im_array,
    y = la_array,
    batch_size = 10, #batch_size代表每次从im_array中的n个图片中选10个来进行训练
    epochs = 8, #操作8次
    validation_split = 0.2, #取训练集中的0.2作为验证集
    shuffle = True
)
model.save("model_v1.h5") #保存模型 
```

这里是进行模型的训练和保存, 编译部分的参数可从上面的文档中查询到, 模型保存的方法也可以百度到



### 图像预测和生成

首先我们取5张图片用上述的方法形成 5 x 512 x 512 x 1 的numpy数组测试样本, 进行预测

```python
model = load_model("model_v1.h5") #加载模型
res=model.predict(te_array,5) #对te_array进行预测,5代表图片个数
```

此时获取到的res数组应该是 float 类型的 5 x 512 x 512 x 1 数组, 此时的res数组代表的是对应像素位置是 0/1 的概率, 因此res是一个概率数组. 由于我们最后生成的图片的每一位是 0~255 的, 所以此时需要转换. 

这里提供两种方法进行转换 : 

- 将res数组所有元素乘以255, 再转换成 int 型,然后使用 img.save 方法进行保存

  ```python
  img = Image.fromarray(im_npp) # im_npp为乘以255后的数组, 这里将数组转换成图片
  img = img.convert("L") # 改变图像模式,黑白为L,彩色为RGB 
  img.save("res/"+str(i)+"_predict_v1.png")
  ```

  注意 : 之所以提到这种方法, 因为我第一次写的时候使用的是这种存储方法, 但是不知道为什么一直报错, 后来发现需要转换图像模式, 所以写在这里给大家提个醒

- 使用skimage.io.imsave 方法, 可直接将 0~1的float数组转换成图片

  ```python
  io.imsave(os.path.join(save_path,"%d_predict_v1.png"%i),im_np)
  ```

  更推荐使用这种方法, 因为更简洁.

生成图片效果展示, 前者为标准的, 后者为我们预测的

![](<https://raw.githubusercontent.com/marcel0637/segmentation_models_Demo/master/markdownsrc/t1.png>)

![](<https://github.com/marcel0637/segmentation_models_Demo/blob/master/markdownsrc/t1_p.png?raw=true>)

此时在较小的数据量下, 大致分割出来了一个轮廓. 

注意 : 由于我们是将0~1的数组乘以255, 所以生成的图像还是灰度图, 而不是黑白图



### 一些疑问

在做完较为简单的版本后, 我产生了一些疑问.

- 按照道理来说二分类问题的标签应该是01分布的才对, 但是为什么我使用0~255分布的标签也能得到具有大概轮廓的图像?

  其实仔细想想, 在进行模型训练的时候, loss函数是关键的部分, 二分类问题使用的loss函数是对预测和标准的概率距离进行度量, 也就是说其实我们最小化该参数, 其实也是最小化两张图片的误差, 所以也是可以达到最后的效果的.

- 为什么训练结果时候的acc非常小, loss非常大?

  我认为应该是当我们使用0 ~ 255分布的标签进行评价的时候, 其相同的部分不如0 ~ 1分布的多, 因此会导致acc非常小 ( 我的代码跑出来大概只有0.2 ) ; 同时loss也由于距离增大, 初始值就是很大的, 最后生成出来的也是很大.

  因此后面设置了阈值.



---

到这里为止, 我们已经实现了对原始图像的初步分割, 但是由于数据集较少, 数据未进行处理等问题, 生成的图像效果不佳, 下面开始写一个升级的版本



### 认识生成器

由于训练的时候数据比较大, 直接加载进入内存可能会超内存, 所以需要采用生成器, 其原理是每次返回一定数量的数据; 比如我model.fit的时候, batch_size是10, 而我的生成器每次返回2个数据, 然后model就用这2个去进行训练,重复调用5次生成器, 这样就实现了10个数据的训练. 

```python
#给定路径,起点和终点,进行图像增强,每次迭代返回一个batch_size的训练图片集和标签集 -> 这里的起点和终点对应下标都是包含在内的!
def train_image_generator(image_path,label_path,st,ed,batch_size,aug = None): 
    nowinx = st #设定初始图片
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            im = Image.open(os.path.join(image_path,str(nowinx)+'.png'))
            tmp_im_array = np.array(im) #图片转numpy数组
            tmp_im_array = tmp_im_array / 255 #对数据进行归一化
            #numpy数组添加一维,为了把二维图片转成三维图片集
            tmp_im_array = tmp_im_array[np.newaxis,:,:] 
            
            lb = Image.open(os.path.join(label_path,str(nowinx)+'.png'))
            tmp_lb_array = np.array(lb) #图片转numpy数组
            tmp_lb_array = tmp_lb_array / 255
            tmp_lb_array[tmp_lb_array > 0.5] = 1
            tmp_lb_array[tmp_lb_array <= 0.5] = 0 #设定阈值将其变成01分布的数组
            tmp_lb_array = tmp_lb_array[np.newaxis,:,:]
            
            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array = tmp_lb_array
            else:
                #将新的图片加入到之前的图片集
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) 
                lb_array = np.concatenate((lb_array,tmp_lb_array),axis=0) 
            
            nowinx = st if nowinx==ed else nowinx+1 #如果遍历到了超出最后一个时,就返回到第一个

        im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.
        lb_array = lb_array[:,:,:,np.newaxis]

        if (aug is not None) and (random.random() > 0.4) : 
            #如果传入了数据增强的生成器,就进行数据增强
            new_array = im_array
            new_array = np.concatenate((new_array,lb_array),axis=3) 
            # 把图像和标签合成一张图片进行增强
            new_array = next(aug.flow(new_array,batch_size = batch_size))
            im_array = new_array[:,:,:,0] # 将图像和标签分离
            lb_array = new_array[:,:,:,1]
            im_array = im_array[:,:,:,np.newaxis] #最后给图片集加一个通道,形成四维.
            lb_array = lb_array[:,:,:,np.newaxis]

        yield(im_array,lb_array) #按批次返回数据
```

根据代码我们可以知道, 除去数据增强的部分, 生成器的大体逻辑就是 : 从给定的图片范围中循环的取出batch_size张图片, 变成numpy数组, 使用yield返回回去 ( yield相当于 转其他函数 , 下次被调用时会从这个位置继续运行 )

注 : 如果没有看懂生成器的使用, 这里提供一个个人觉得不错的博客

[生成器的使用参考链接](https://blog.csdn.net/learning_tortosie/article/details/85243310)



### 数据增强

其实数据增强既可以自己实现, 也可以调用现有的类, 这里调用ImageDataGenerator来实现数据增强

首先定义一个数据增强生成器

```python
aug = ImageDataGenerator( #定义一个数据增强生成器
    rotation_range = 0.05, # 定义旋转范围
    zoom_range = 0.05, # 按比例随机缩放图像尺寸
	width_shift_range = 0.05, # 图片水平偏移幅度
    height_shift_range = 0.05, # 图片竖直偏移幅度
    shear_range = 0.05, # 水平或垂直投影变换
	horizontal_flip = True, # 水平翻转图像
    fill_mode = "reflect" # 填充像素,出现在旋转或平移之后
)
```

具体参数同样给出文档供参考

[ImageDataGenerator 参考链接](https://keras.io/zh/preprocessing/image/)

生成器的使用 : aug.flow(数据,size) 可以返回出一个生成器, 生成器每次取出经过增强后的数据size个, 每次调用next即可获取一份. 

**注意** : 我们在进行数据增强的时候, 一定是图像和标签同时进行增强的. 实现有两种方法

- 方法1 : 为图像设置一个生成器, 标签设置一个生成器, 然后赋予他们同样的seed, 这样他们就会进行相同变化

  ​	     官方文档参考同上的 ImageDataGenerator

- 方法2 : 将标签作为图像的第2个通道, 进行合并, 然后对图像进行数据增强后, 再将图像和标签分离开来

  ​	     这里采用的是方法2



### 模型训练

这里使用的模型训练方法同样使用了生成器, 具体参数也是可以在第一次训练模型提供的文档中找到

```python
train_gen = train_image_generator("image","label",0,20,4,aug) # 获取一个训练数据生成器
validate_gen = train_image_generator("image","label",21,29,3,None) # 获取一个验证数据生成器
#定义并编译一个模型

model = Unet('resnet34', input_shape = (512, 512, 1), encoder_weights = None)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

#进行模型的训练,并用his来记录训练过程中的参数变化,方便最后生成图像
his = model.fit_generator( 
    generator = train_gen, #训练集生成器
    steps_per_epoch = 300, #训练集每次epoch的数量
    validation_data = validate_gen, #验证集生成器
    validation_steps = 3, #验证集每次epoch的数量
    epochs = 10 #进行epoch的次数
)
```

这里也有一些地方可以探讨一下.

- 我想像model.fit中一样从训练集中随机抽出一些作为验证集怎么办?

  ​	这个问题我在网上找了很久, 并没有发现相关的方法, 可能想要实现的话需要自己手写.

- model里面的 epochs, steps_per_epoch, validation_steps, 生成器里面的 batch_size之间的关系

  ​	我理解的大概逻辑是这样的 : 模型进行epochs轮训练, 每轮总共会使用steps_per_epoch的训练集, 而这些训练集也不是一次加载到内存的, 而是每次使用就从生成器中取出batch_size个, 而validation_steps也是同理.

  ​	以我上面代码作为举例 : 模型进行10轮训练, 每轮使用300个训练图片, 分75次, 每次从生成器中取出4个进行训练, 训练满300个后再进行验证



最后进行模型保存, 进行模型评估, 这里评估的代码都是在网上找的, 就不分析了, 很容易找到. 

这是我训练过程中的图片

![](<https://github.com/marcel0637/segmentation_models_Demo/blob/master/markdownsrc/plot_v4.png?raw=true>)

### 使用生成器的图像预测和生成

测试图片的处理其实大体是同载入训练图片的步骤相同, 只是要同样记得设置阈值, 代码就不贴了.

然后就是生成图像, 生成图像的时候也是要记得设置阈值, 这里贴一下代码

```python
def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile): #取出预测结果中的每一个
        im_np = item[:,:,0]
        im_np[im_np > 0.5] = 1
        im_np[im_np <= 0.5] = 0
        io.imsave(os.path.join(save_path,"%d_predict_v4.png"%i),im_np)
```

到这里大概就完成了整个项目.

---



### 最后再提出一些点

- 关于阈值

  ​	阈值的作用其实是帮助我们将标签进行分类, 从而得到更好的结果. 

  ​	我们一定需要对训练使用的标签进行设置阈值, 这样才能在对模型进行评估的时候得到正确的 acc . 此外, 这里提一下, 由于标签的图片本身就是近似于黑白的, 白色基本就是 255,254, 黑色基本就是 0,1, 所以对训练的标签进行阈值的调参感觉没有必要, 我也尝试过, 确实没什么效果上的差异.

  ​	对于进行图像生成的时候, 设置的阈值参数可以多次尝试, 然后择优. (不设置的话会使图片出现很多灰团)

- 建议将模型的训练和最后的预测分开进行操作, 这样可以把模型保存起来, 以免运行到预测的时候报错, 前面训练了的模型又得重新训练

- 如果自己实践的模型预测出来是全黑/全白, 一定先理一下自己代码的逻辑有没有问题, 看看训练的图片和进行测试图片是否进行的操作相同. 其次, 阈值的设置与否, 大小的选取也有可能会导致全黑/全白.

  比如我最开始用0~255的图片进行训练, 然后我把预测出来的标签进行阈值的设置, 这个时候由于训练出来的图片对比度很低, 灰白很相近, 阈值不恰当就导致了出现全白的情况.

- 一开始我在自己实践的时候, 为了快速检验代码是否正确, 把epochs和steps_per_epoch设置得很小, 结果出现了测试结果近乎全白或者效果很差的问题, 所以自己在实践的时候选取一个合适的epochs和steps_per_epoch.

  ---- 这里吹一下Google 的免费 GPU , 我最后的代码10个epochs和300的steps_per_epoch只花10~20分钟就跑出来了, 不过使用起来没想象中的顺手, 建议把代码写好后直接丢上去运行, 修改后再次上传可能会等很久才能更新出来. 

- 在训练中是否使用数据增强, 验证集都是可以选择的. 只是效果不同 ( 可能会降低acc, 但是效果会好 )

- 我的代码最后达到的acc是93%左右, 其实还有很多地方可以调参, 可以进行优化, 由于本人只是通过该项目理清一下操作的流程和代码逻辑, 所以就没有花时间继续优化, 大家可以花时间进行优化, 以达到更好的效果

- 我在实践的时候写了很多个版本的代码, 每次都做了一些微调, 并保存了下来, 预测出来的图片可以供大家比较一下效果

- 我最后预测出来的图片有一些小黑点, 调阈值也比较难去除, 希望有人知道如何解决的话可以留言.



---

**最后** , 我的整个项目代码和使用Colab用到的操作放在了我的github上. 如果文章有错请留言指出, 谢谢!

[我的github链接](https://github.com/marcel0637/segmentation_models_Demo)




