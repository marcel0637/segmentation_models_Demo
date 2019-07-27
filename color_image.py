import cv2
import random

img = cv2.imread('./res/0_predict_v4.png')
h, w, _= img.shape

#进行二值化预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

c_max = []
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
 
    # 对小的轮廓进行特殊处理.
    if(area < (h/10*w/10)):
        c_min = []
        c_min.append(cnt)
        # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
        cv2.drawContours(img, c_min, -1, (255,255,255), thickness=-1)
        continue
    #
    c_max.append(cnt)

# option 1 进行图片轮廓内部颜色填充
for i in range(len(c_max)):
    cc = []
    #由于必须绘制 轮廓内 必须传递list,所以这里创建了一个
    cc.append(c_max[i])
    cv2.drawContours(img, cc, -1, (random.randint(0,255), random.randint(0,255),
         random.randint(0,255)), thickness=-1)

# option 2 不进行轮廓内部填充,直接对轮廓线进行绘制(达到去除图片黑点的效果)
#cv2.drawContours(img, c_max, -1, (0, 0, 0), 1)
#cv2.imwrite("img.png", img)  
cv2.imshow("img", img)
cv2.waitKey(0)