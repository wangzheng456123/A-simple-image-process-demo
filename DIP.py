# -*- coding: utf-8 -*-
import numpy as np

import matplotlib
import matplotlib.image as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from math import pow, pi
import sys

def _rgb2gray(A):
    if len(A.shape) < 3:
        return A
    x = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            x.append(np.float(A[i][j][0] * 0.3 + A[i][j][1] * 0.59 + A[i][j][2] * 0.11))
    return np.array(x).reshape((A.shape[0], A.shape[1]))

#*********************************************************************************************
#灰度变换
#参数意义：
#    img：输入图像的numpy数组。
#    f：对图像进行灰度变幻的函数，定义域和值域都必须[0, 1]。
#    返回值：变换后后图像，数据类型为numpy数组。
#    note：此变换还未被加入到图形界面中，使用请直接调用。
#*********************************************************************************************
def _Intensity_Transform(img, f):
    new_img = np.zeros(img.shape)
    for i in img.shape[0]:
        for j in img.shape[1]:
            new_img[i][j] = f(img[i][j])
    return new_img

#*********************************************************************************************
#卷积：
#参数意义：
#   img：输入图像的numpy数组。函数会对输入图像进行0填充。
#   filter_H：卷积核的高度。必须为奇数，默认为3。
#   filter_H：卷积核的宽度。必须为奇数，默认为3。
#   filter：自定义的卷积核，可以是列表或者一维的numpy数组。
#   _type：函数中有两个内置的卷积核，分别为方差为1的高斯卷积核以及拉普拉斯卷积核，相应的变量值为'Gaussian'和'Laplacian'。
#   返回值：为卷积所得图像的numpy数组。
#*********************************************************************************************
def _Conv(img, filter_h = 3, filter_w = 3, filter = [], _type = 'None'):
    H, W = img.shape
    center = (int(filter_h / 2.0), int(filter_w / 2.0))  
    pad_img = np.zeros((img.shape[0] + 2 * center[0], img.shape[1] + 2 * center[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pad_img[i + center[0]][j + center[1]] = img[i][j]
    new_img = np.zeros(img.shape)
    if _type == 'None':
        filter = np.array(filter)
        filter = filter.reshape(filter_h, filter_w)
    if _type == 'Gaussian':
        filter = np.zeros((filter_h, filter_w))
        for i in range(filter_h):
            for j in range(filter_w):
                x = i - center[0]
                y = j = center[1]
                filter[i][j] = np.exp(-1 * (pow(x, 2) + pow(y, 2)) / 2) / 2 * pi
        s = np.sum(filter)
        for i in range(filter_h):
            for j in range(filter_w):
                x = i - center[0]
                y = j = center[1]
                filter[i][j] = filter[i][j] / s
    if _type == 'Laplacian':
        filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cur = pad_img[i: i + filter_h,
                        j: j + filter_w]
            cur = cur * filter
            conv_sum = np.sum(cur)
            new_img[i][j] = conv_sum
    return new_img

#*********************************************************************************************
#直方图均衡：
#参数意义：
#    img：输入图像的numpy数组，灰度值为0，1之间的连续值。 
#    _range：灰度值范围。若不加声明，默认为0 - 255。
#    返回值：为进行直方图匹配处理后图像的numpy数组，灰度值为0，1之间的连续值。
#*********************************************************************************************
def _Histogram_Equalization(img, _range = 255):
    img = img * (_range - 1)
    img = np.round(img)
    img = img.astype(int)
    num_pix = img.shape[0] * img.shape[1]
    hist = np.zeros(_range)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1.0
    new_hist = np.zeros(_range)        
    for i in range(len(hist)):
        for j in range(i + 1):
            new_hist[i] += hist[j]
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = hist[img[i][j]] / num_pix
    return new_img 

#*********************************************************************************************
#直方图匹配：
#参数意义：
#    img：输入图像的numpy数组，灰度值为0到1之间的连续值。 
#    img_ref:参考图像的numpy数组，灰度值为0到1之间的连续值。 
#    _range：灰度值范围。若不加声明，默认为0 - 255。
#    返回值：匹配后图像的numpy数组，灰度值为0，1之间的连续值。
#*********************************************************************************************
def _Histogram_Matching(img, img_ref, _range = 255):
    img = img * (_range - 1)
    img = np.round(img)
    img = img.astype(int)
    img_ref = img_ref * (_range - 1)
    img_ref = np.round(img_ref)
    img_ref = img_ref.astype(int)
    hist_o = np.zeros(_range)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist_o[img[i][j]] += 1.0
    new_hist_o = np.zeros(_range) 
    for i in range(len(hist_o)):
        for j in range(i + 1):
            new_hist_o[i] += hist_o[j]
    hist_g = np.zeros(_range)
    for i in range(img_ref.shape[0]):
        for j in range(img_ref.shape[1]):
            hist_g[img_ref[i][j]] += 1.0
    new_hist_g = np.zeros(_range)
    for i in range(len(hist_g)):
        for j in range(i + 1):
            new_hist_g[i] += hist_g[j]
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = 0
            while k < _range and new_hist_g[k] < new_hist_o[img[i][j]]:
                k += 1
            new_img[i][j] = k
    new_img = new_img / 255
    return new_img

#*********************************************************************************************
#快速傅里叶变换：
#参数意义：
#    img：输入图像的numpy数组。
#    返回值：图像傅里叶变换的结果，数据类型为numpy数组。
#*********************************************************************************************
def _FFT(img):
    f = np.fft.fft2(img)                              
    fshift = np.fft.fftshift(f) 
    return fshift

#*********************************************************************************************
#快速傅里叶反变换：
#参数意义：
#    img：输入图像的numpy数组。
#    返回值：由傅里叶变换结果还原出的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Reverse_FFT(img):
    ishift = np.fft.ifftshift(img)
    img = np.fft.ifft2(ishift)
    return np.abs(img)

#*********************************************************************************************
#低通滤波器
#参数意义：
#    img：输入图像的numpy矩阵。
#    _type：滤波器类型，有高斯，巴特沃斯，以及理想滤波器三种。相应的参数值为Gaussian，Butterworth和Ideal。
#    cut_off：滤波器的截止频率。
#    order：滤波器的阶数，仅当滤波器的类型为巴特沃斯时有意义。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Low_Pass_Filter(img, _type, cut_off, order = 0):
    f = np.fft.fft2(img)                              
    fshift = np.fft.fftshift(f)  
    result = 20 * np.log(np.abs(fshift))
    center = (img.shape[0] / 2, img.shape[1] / 2)
    filter = np.zeros(img.shape)
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            D = np.sqrt(pow(i - center[0], 2) + pow(j - center[1], 2))
            if _type == 'Gaussian':
                filter[i, j] = np.exp(-1 * pow(D, 2) / (2 * pow(cut_off, 2)))
            if _type == 'Butterworth':
                filter[i, j] = 1 / (1 + pow(D / cut_off, 2 * order))
            if _type == 'Ideal':
                if D > cut_off:
                    filter[i, j] = 0
                if D <= cut_off:
                    filter[i, j] = 1
    res = fshift * filter
    ishift = np.fft.ifftshift(res)
    io = np.fft.ifft2(ishift)
    return np.abs(io)

#*********************************************************************************************
#高通滤波器
#参数意义：
#    img：输入图像的numpy矩阵。
#    _type：滤波器类型，有高斯，巴特沃斯，以及理想滤波器三种。相应的参数值为Gaussian，Butterworth和Ideal。
#    cut_off：滤波器的截止频率。
#    order：滤波器的阶数，仅当滤波器的类型为巴特沃斯时有意义。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _High_Pass_Filter(img, _type, cut_off, order = 0):
    f = np.fft.fft2(img)                              
    fshift = np.fft.fftshift(f)  
    result = 20 * np.log(np.abs(fshift))
    center = (img.shape[0] / 2, img.shape[1] / 2)
    filter = np.zeros(img.shape)
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            D = np.sqrt(pow(i - center[0], 2) + pow(j - center[1], 2))
            if _type == 'Gaussian':
                filter[i, j] = 1 - np.exp(-1 * pow(D, 2) / (2 * pow(cut_off, 2)))
            if _type == 'Butterworth':
                filter[i, j] = 1 - 1 / (1 + pow(D / cut_off, 2 * order))
            if _type == 'Ideal':
                if D > cut_off:
                    filter[i, j] = 1
                if D <= cut_off:
                    filter[i, j] = 0
    res = fshift * filter
    ishift = np.fft.ifftshift(res)
    io = np.fft.ifft2(ishift)
    return np.abs(io)

#*********************************************************************************************
#随机噪声
#参数意义：
#    img：输入图像的numpy数组，灰度值为0到1之间的连续值。
#    _type：噪声类型。根据噪声类型的不同，arg1和arg2有不同的含义，下面将逐一介绍。
#    _type = 'Gaussian'：添加服从高斯分布的噪声。arg1表示噪声的均值，arg2表示噪声的方差。
#    _type = 'Gamma'：添加服从伽马分布的噪声。arg1和arg2分别为其形状参数和逆尺度参数。
#    _type = 'Uniform'：添加服从[arg1, arg2]上均匀分布的噪声。
#    _type =  'Pepper_Salt'：椒盐噪声。arg1和arg2的值需介于0和1之间。
#    会在图像的每个像素上以arg1的概率产生盐噪声，1 - arg2的概率产生椒噪声。
#    返回值：加入噪音后图像，数据类型为numpy数组。
#*********************************************************************************************
def _Random_Noise(img, _type, arg1 = 0, arg2 = 1):
    noise_img = np.zeros(img.shape)
    if _type == 'Gaussian':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                noise_img[i][j] = img[i][j] + np.random.normal(arg1, arg2)
                if noise_img[i][j] > 1:
                    noise_img[i][j] = 1.0
                if noise_img[i][j] < 0:
                    noise_img[i][j] = 0.0
    if _type == 'Gamma':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                noise_img[i][j] = img[i][j] + np.random.gamma(arg1, arg2)
                if noise_img[i][j] > 1:
                    noise_img[i][j] = 1.0
                if noise_img[i][j] < 0:
                    noise_img[i][j] = 0.0
    if _type == 'Uniform':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                noise_img[i][j] = img[i][j] + np.random.uniform(arg1, arg2)
                if noise_img[i][j] > 1:
                    noise_img[i][j] = 1.0
                if noise_img[i][j] < 0:
                    noise_img[i][j] = 0.0
    if _type == 'Pepper_Salt':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rand = np.random.uniform(0, 1)
                if rand < arg1:
                    noise_img[i][j] = 0.0
                    continue
                if rand > arg2:
                    noise_img[i][j] = 1.0
                    continue
                noise_img[i][j] = img[i][j]         
    return noise_img

#*********************************************************************************************
#周期噪声
#参数意义：
#    img：输入图像的numpy数组。
#    f_x：x方向频率。
#    f_y：y方向频率。
#    B_x：x方向相位，默认为0。
#    B_y：y方向相位，默认为0。
#    A：振幅，默认为0.5。
#    返回值：加入噪音后图像，数据类型为numpy数组。
#*********************************************************************************************
def _Periodic_Noise(img, f_x, f_y, B_x = 0, B_y = 0, A = 0.5):
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = 2 * pi * f_y * (i + B_y) / img.shape[0] + 2 * pi * f_x * (j + B_x) / img.shape[1]
            new_img[i][j] = img[i][j] + A * np.sin(val)
            if new_img[i][j] > 1:
                new_img[i][j] = 1
            if new_img[i][j] < 0:
                new_img[i][j] = 0
    return new_img

#*********************************************************************************************
#非适应性空域滤波器
#参数意义：
#    img：输入图像的numpy数组，灰度值为0到1之间的连续值。
#    _type：滤波器的类型，可以为以下几种类型之一：
#    _type = 'Arithmetic_Mean'：算数均值滤波器。
#    _type = 'Geometric_Mean'： 几何均值滤波器。
#    _type = 'Harmonic_Mean'： 调和均值滤波器。
#    _type = 'Contrharmonic_Mean'： 谐波滤波器 
#    _type = 'Median'：中值滤波器。
#    _type =  'Max'：极大值滤波器。
#    _type =  'Min'：极小值滤波器。 
#    _type =  'Midpoint'：中点滤波器。
#    filter_w，filter_h：滤波器的宽度和长度。必须为奇数。
#    Q：仅当滤波器为谐波滤波器时有意义，默认为0。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Filter_in_Space(img, _type, filter_h, filter_w, Q = 0):
    center = (int(filter_h / 2), int(filter_w / 2))
    pad_img = np.zeros([int(img.shape[0] + 2 * center[0]), int(img.shape[1] + 2 * center[1])])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pad_img[i + center[0]][j + center[1]] = img[i][j]
    new_img = np.zeros(img.shape)
    if _type == 'Arithmetic_Mean':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                new_img[i][j] = np.sum(cur) / (filter_w * filter_h)
    if _type == 'Geometric_Mean':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i][j] = 1
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                for r in cur:
                    for element in r:
                        new_img[i][j] *= element
                new_img[i][j] = pow(new_img[i][j], 1 / (filter_w * filter_h))
    if _type == 'Harmonic_Mean':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i][j] = 0
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                for r in cur:
                    for element in r:
                        if element == 0:
                            new_img[i][j] += 100000
                            continue
                        new_img[i][j] += 1 / element
                new_img[i][j] = filter_w * filter_h / new_img[i][j]
    if _type == 'Contraharmonic_Mean':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                tem = 0
                new_img[i][j] = 0
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                for r in cur:
                    for element in r:
                        if Q < 0 and element == 0:
                            new_img[i][j] = 100000
                            tem = 100000
                            continue
                        new_img[i][j] += pow(element, Q + 1)
                        tem += pow(element, Q)
                if tem == 0:
                    new_img[i][j] = 1
                    continue
                new_img[i][j] = new_img[i][j] / tem
    if _type == 'Median':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                cur = cur.reshape(filter_w * filter_h)
                cur = np.sort(cur)
                if len(cur) % 2:
                    new_img[i][j] = cur[int(len(cur) / 2 + 1)]
                if not len(cur) % 2:
                    new_img[i][j] = (cur[int(len(cur) / 2  - 1)] + cur[(len(cur) / 2)]) / 2
    if _type == 'Max':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                cur = cur.reshape(filter_w * filter_h)
                new_img[i][j] = max(cur)
    if _type == 'Min':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                cur = cur.reshape(filter_w * filter_h)
                new_img[i][j] = min(cur)
    if _type == 'Midpoint':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                cur = cur.reshape(filter_w * filter_h)
                new_img[i][j] = (min(cur) + max(cur)) / 2
    return new_img

#*********************************************************************************************
#适应性空域滤波器
#参数意义：
#    img：输入图像的numpy数组，灰度值为0到1之间的连续值。
#    _type：滤波器的类型，可以为以下几种两类之一：
#    _type = 'Local_Noise_Reduction'：局部降噪滤波器。
#    _type = 'Adaptive_Median'： 适应性中值滤波器。
#    filter_w，filter_h：滤波器的宽度和长度，必须为奇数。
#    variance：使用局部降噪滤波器时有意义，为噪音的方差。
#    filter_h_max，filter_w_max：使用适应性中值滤波器时有意义，为可扩展的最大滤波器的宽度和高度。必须为奇数。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Adaptive_Filter_in_Space(img, _type, filter_w, filter_h, variance = 0, filter_h_max = 0, filter_w_max = 0):
    new_img = np.zeros(img.shape)
    if _type == 'Local_Noise_Reduction':
        center = (int(filter_w / 2), int(filter_h / 2))
        pad_img = np.zeros([int(img.shape[0] + 2 * center[0]), int(img.shape[1] + 2 * center[1])])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pad_img[i + center[0]][j + center[1]] = img[i][j]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cur = pad_img[i:i + filter_h, j:j + filter_w]
                var_c = cur.var()
                mean_c = cur.mean()
                if var_c != 0:
                    t = variance / var_c
                else:
                    t = 1
                if t > 1:
                    t = 1
                new_img[i][j] = img[i][j] - t * (img[i][j] - mean_c)
    if _type == 'Adaptive_Median':
        H = filter_h
        W = filter_w
        center = (int(filter_h_max / 2), int(filter_w_max / 2))
        pad_img = np.zeros([int(img.shape[0] + 2 * center[0]), int(img.shape[1] + 2 * center[1])])
        print(pad_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pad_img[i + center[0]][j + center[1]] = img[i][j]
        print(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dif_h = int(filter_h_max / 2) - int(filter_h / 2)
                dif_w = int(filter_w_max / 2) - int(filter_w / 2)
                cur = pad_img[i + dif_h:i + dif_h + filter_h, j + dif_w:j + dif_w + filter_w]
                cur = cur.reshape(filter_h * filter_w)
                max_c = max(cur)
                min_c = min(cur)
                cur = np.sort(cur)
                if len(cur) % 2:
                    median_c = cur[int(len(cur) / 2 + 1)]
                if not len(cur) % 2:
                    median_c = (cur[int(len(cur) / 2  - 1)] + cur[(len(cur) / 2)]) / 2
                while not (min_c < median_c and median_c < max_c or filter_h == filter_h_max):
                    filter_h += 2
                    filter_w += 2
                    dif_h = int(filter_h_max / 2) - int(filter_h / 2)
                    dif_w = int(filter_w_max / 2) - int(filter_w / 2)
                    cur = pad_img[i + dif_h:i + dif_h + filter_h, j + dif_w:j + dif_w + filter_w]
                    cur = cur.reshape(filter_h * filter_w)
                    max_c = max(cur)
                    min_c = min(cur)
                    np.sort(cur)
                    if len(cur) % 2:
                        median_c = cur[int(len(cur) / 2 + 1)]
                    if not len(cur) % 2:
                        median_c = (cur[int(len(cur) / 2  - 1)] + cur[(len(cur) / 2)]) / 2
                if filter_h == filter_h_max:
                    new_img[i][j] = median_c
                else:
                    if min_c < img[i][j] and img[i][j] < max_c:
                        new_img[i][j] = img[i][j]
                    else:
                        new_img[i][j] = median_c
                filter_h = H
                filter_w = W
    return new_img

#*********************************************************************************************
#运动模糊
#参数意义：
#    img：输入图像的numpy数组。
#    a：x方向速度。
#    b：y方向速度。
#    T：运动的时间。
#    返回值：退化的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Motion_Degradation(img, a, b, T):
    f = np.fft.fft2(img)                              
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if i == 0 and j == 0:
                f[i][j] = T
                continue
            v = (i * a + j * b) * pi
            f[i][j] *= (T / v) * np.sin(v) * np.exp(-1.j * v)
    new_img = np.fft.ifft2(f)
    return np.abs(new_img)

#*********************************************************************************************
#陷通滤波器
#参数意义：
#    img：输入图像的numpy数组。
#    _type：滤波器的类型，可以为以下几种之一：
#    _type = 'Gaussian'：高斯滤波器。
#    _type = 'Butterworth'： 巴特沃斯滤波器。
#    _type = 'Ideal'：理想滤波器
#    cut_off：截止频率。
#    order：阶数。仅当滤波器类型为巴特沃斯时有意义。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Notch_Pass_Filter(img, _type, u, v, cut_off, order = 0):
    f = np.fft.fft2(img)                              
    fshift = np.fft.fftshift(f)  
    center = (img.shape[0] / 2, img.shape[1] / 2)
    filter = np.zeros(img.shape)
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            D = np.sqrt(pow(i - center[0] + v, 2) + pow(j - center[1] + u, 2))
            D_n = np.sqrt(pow(i - center[0] - v, 2) + pow(j - center[1] - u, 2))
            if _type == 'Gaussian':
                filter[i][j] = np.exp(-1 * pow(D, 2) / (2 * pow(cut_off, 2)))
                filter[i][j] *= np.exp(-1 * pow(D_n, 2) / (2 * pow(cut_off, 2)))
            if _type == 'Butterworth':
                filter[i][j] = 1 / (1 + pow(D / cut_off, order))
                filter[i][j] *= 1 / (1 + pow(D_n / cut_off, order))
            if _type == 'Ideal':
                if D > cut_off and D_n > cut_off:
                    filter[i][j] = 0
                else:
                    filter[i][j] = 1
    res = fshift * filter
    ishift = np.fft.ifftshift(res)
    io = np.fft.ifft2(ishift)
    return np.abs(io)

#*********************************************************************************************
#陷阻滤波器
#参数意义：
#    img：输入图像的numpy数组。
#    _type：滤波器的类型，可以为以下几种之一：
#    _type = 'Gaussian'：高斯滤波器。
#    _type = 'Butterworth'： 巴特沃斯滤波器。
#    _type = 'Ideal'：理想滤波器
#    cut_off：截止频率。
#    order：阶数。仅当滤波器类型为巴特沃斯时有意义。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Notch_Reject_Filter(img, _type, u, v, cut_off, order = 0):
    f = np.fft.fft2(img)                              
    fshift = np.fft.fftshift(f)  
    center = (img.shape[0] / 2, img.shape[1] / 2)
    filter = np.zeros(img.shape)
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            D = np.sqrt(pow(i - center[0] + v, 2) + pow(j - center[1] + u, 2))
            D_n = np.sqrt(pow(i - center[0] - v, 2) + pow(j - center[1] - u, 2))
            if _type == 'Gaussian':
                filter[i][j] = 1 - np.exp(-1 * pow(D, 2) / (2 * pow(cut_off, 2)))
                filter[i][j] *= 1 - np.exp(-1 * pow(D_n, 2) / (2 * pow(cut_off, 2)))
            if _type == 'Butterworth':
                filter[i][j] = 1 - 1 / (1 + pow(D / cut_off, order))
                filter[i][j] *= 1 - 1 / (1 + pow(D_n / cut_off, order))
            if _type == 'Ideal':
                if D > cut_off and D_n > cut_off:
                    filter[i][j] = 1
                else:
                    filter[i][j] = 0
    res = fshift * filter
    ishift = np.fft.ifftshift(res)
    io = np.fft.ifft2(ishift)
    return np.abs(io)

#*********************************************************************************************
#最佳陷通滤波器
#参数意义：
#    img：输入图像的numpy数组。
#    filter_w，filter_h：窗口的宽度和长度，必须为奇数。
#    u，v：指定的要去除的x方向和y方向上的频率。
#    cut_off：截止频率，默认为5。
#    order：阶数，默认为2。
#    返回值：滤波后的图像，数据类型为numpy数组。
#*********************************************************************************************
def _Optimum_Notch_Filter(img, filter_h, filter_w, u, v, cut_off = 5, order = 2):
    center = (int(filter_h / 2.0), int(filter_w / 2.0))  
    pad_img = np.zeros((img.shape[0] + 2 * center[0], img.shape[1] + 2 * center[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pad_img[i + center[0]][j + center[1]] = img[i][j]
    noise = _Notch_Pass_Filter(img, 'Butterworth', u, v, cut_off, order)
    pad_noise = np.zeros((img.shape[0] + 2 * center[0], img.shape[1] + 2 * center[1]))
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            pad_noise[i + center[0]][j + center[1]] = noise[i][j]
    new_img = np.zeros(img.shape)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            cur = pad_img[i:i + filter_h, j:j + filter_w]
            cur = cur.reshape(filter_h * filter_w)
            cur_n = pad_noise[i:i + filter_h, j:j + filter_w]
            cur_n = cur_n.reshape(filter_h * filter_w)
            mean_n = cur_n.mean()
            mean_c = cur.mean()
            mean_nn = (cur_n * cur).mean()
            mean_s = np.array([pow(num, 2) for num in cur_n]).mean()
            w = (mean_nn - mean_n * mean_c) / (mean_s - pow(mean_n, 2))
            new_img[i][j] = img[i][j] - w * noise[i][j]
    return new_img

#*********************************************************************************************
#哈尔快速小波变换：
#参数意义：
#    seq：输入参数
#    返回值：哈尔快速小波变换各层的输出，数据类型为列表。
#*********************************************************************************************
def _Haar_Fast_Wavelet(seq):
    Haar_scale = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    Haar_wavelet = [1 / np.sqrt(2), -1 / np.sqrt(2)]
    out = []
    state = seq
    res = []
    while len(state) != 1:
        tem = []
        for i in range(int(len(state) / 2)):
            tem.append(Haar_scale[0] * state[2 * i] + Haar_scale[1] * state[2 * i + 1])
            out.append(Haar_wavelet[0] * state[2 * i] + Haar_wavelet[1] * state[2 * i + 1])
        res += out
        out = []
        state = tem
    return res + state

#*********************************************************************************************
#哈尔快速小波反变换：
#参数意义：
#    seq：输入参数
#    返回值：哈尔快速小波反变换各层的输出，数据类型为列表。
#*********************************************************************************************
def _Reverse_Haar_Fast_Wavelet(seq):
    Haar_scale = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    Haar_wavelet = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
    inp = np.array([seq[1]])
    state = np.array([seq[0]])
    last = 1
    while len(state) != len(seq):
        lenth = len(inp)
        for i in range(lenth):
            inp = np.insert(inp, 2 * i + 1, 0)
            state = np.insert(state, 2 * i + 1, 0)
        inp = np.insert(inp, 0, 0)
        inp = np.insert(inp, len(inp), 0) 
        state = np.insert(state, 0, 0)
        state = np.insert(state, len(state), 0)
        tem = []
        cur = []
        for i in range(len(inp) - 2):
            t = 0
            t += Haar_scale[0] * state[i] + Haar_scale[1] * state[i + 1] 
            t += Haar_wavelet[0] * inp[i] + Haar_wavelet[1] * inp[i + 1]
            tem.append(t)
            if last < len(seq) - 1:
                last += 1
                cur.append(seq[last])
        inp = cur
        state = tem
    return state
    
# GUI代码

import tkinter as tk
import tkinter.messagebox as tkm
import tkinter.filedialog


outfile = './output'

win = tkinter.Tk()
win.title("数字图像处理")
win.geometry("1000x800")

f = plt.figure()
fig1 = plt.subplot(1,1,1)
canvas = FigureCanvasTkAgg(f, win)

def Choose_File():
    fig1.clear()
    select_file = tkinter.filedialog.askopenfilename(title = '选择图片')
    global img
    global H, W, C
    img = mp.imread(select_file, 0)
    H, W, C = img.shape
    img = _rgb2gray(img)
    tem = img.reshape(H * W)
    _max = max(tem)
    for i in range(H):
        for j in range(W):
            if _max != 0:
                img[i][j] /= _max
            else: img[i][j] = 0   
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)

def _Get_Argument(res, e, son):
    for entry in e:
        res.append(entry.get())
    son.quit()
    son.destroy()
    
def _Choose_Ref(ref_img):
    select_file = tkinter.filedialog.askopenfilename(title = '选择图片')
    img = mp.imread(select_file, 0)
    img = _rgb2gray(img)
    for i in range(img.shape[0]):
        ref_img.append([])
        for j in range(img.shape[1]):
            ref_img[i].append(img[i][j])
    
def Histogram_Equalization():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    tk.Label(son, text= "像素范围：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    res = []
    button = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res ,e, son))
    button.pack()
    son.mainloop()
    _range = 255
    if len(res[0]) != 0:
        _range = int(res[0])
    img = _Histogram_Equalization(img, _range)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
                
def Histogram_Matching():  
    global img
    fig1.clear()
    ref_img = []
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "像素范围：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    button2 = tkinter.Button(son, text = "选择参考图像", command = lambda: _Choose_Ref(ref_img))
    button2.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _range = 255
    if len(res[0]) != 0:
        _range = int(res[0])
    ref_img = np.array(ref_img)
    img = _Histogram_Matching(img, ref_img, _range)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Conv():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "卷积核：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "卷积核高度：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "卷积核宽度：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "类型：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    kernel = []
    if len(res[0]) != 0:
        kernel = res[0].split(',')
    for i in range(len(kernel)):
        kernel[i] = float(kernel[i])
    filter_h = 3
    if len(res[1]) != 0:
        filter_h = int(res[1])
    filter_w = 3
    if len(res[2]) != 0:
        filter_w = int(res[2])
    _type = res[3]
    if len(_type) == 0:
        _type = 'None'
    img = _Conv(img, filter_h = filter_h, filter_w = filter_w, filter = kernel, _type = _type)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def FFT():
    global img
    fig1.clear()
    img = _FFT(img)
    tem = 20 * np.log(np.abs(img))
    fig1.imshow(tem, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Reverse_FFT():
    global img
    fig1.clear()
    img = _Reverse_FFT(img)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)

def High_Pass_Filter():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "滤波器类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "截止频率：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "阶数：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    cut_off = int(res[1])
    order = 0
    if _type == 'Butterworth':
        orer = int(res[2])
    img = _High_Pass_Filter(img, _type, cut_off, order = order)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Low_Pass_Filter():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "滤波器类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "截止频率：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "阶数：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    cut_off = int(res[1])
    order = 0
    if _type == 'Butterworth':
        orer = int(res[2])
    img = _Low_Pass_Filter(img, _type, cut_off, order = order)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)   
    
def Random_Noise():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "噪声类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "参数1：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "参数2：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    arg1 = float(res[1])
    arg2 = float(res[2])
    order = 0
    img = _Random_Noise(img, _type, arg1, arg2)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Periodic_Noise():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x300")
    e = []
    res = []
    tk.Label(son, text= "x频率：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "y频率：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "x相位：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "y相位：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    tk.Label(son, text= "幅度：").pack()
    e5 = tk.Entry(son)
    e.append(e5)
    e5.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    f_x = float(res[0])
    f_y = float(res[1])
    B_x = 0
    if len(res[2]) != 0:
        B_x = float(res[2])
    B_y = 0
    if len(res[3]) != 0:
        B_y = float(res[3])
    A = 0.5
    if len(res[4]) != 0:
        A = float(res[4])
    order = 0
    img = _Periodic_Noise(img, f_x, f_y, B_x = B_x, B_y = B_y, A = A)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Filter_in_Space():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "卷积核高度：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "卷积核宽度：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "Q：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    filter_h = int(res[1])
    filter_w = int(res[2])
    Q = 0
    if len(res[3]) != 0:
        Q = int(res[3])
    img = _Filter_in_Space(img, _type, filter_h, filter_w, Q)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Adaptive_Filter_in_Space():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x400")
    e = []
    res = []
    tk.Label(son, text= "类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "卷积核高度：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "卷积核宽度：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "方差：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    tk.Label(son, text= "最大卷积核高度：").pack()
    e5 = tk.Entry(son)
    e.append(e5)
    e5.pack()
    tk.Label(son, text= "最大卷积核宽度：").pack()
    e6 = tk.Entry(son)
    e.append(e6)
    e6.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    filter_h = int(res[1])
    filter_w = int(res[2])
    var = 0
    if len(res[3]) != 0:
        var = float(res[3])
    filter_h_max = 0
    if len(res[4]) != 0:
        filter_h_max = int(res[4])
    filter_w_max = 0
    if len(res[5]) != 0:
        filter_w_max = int(res[5])
    img = _Adaptive_Filter_in_Space(img, _type, filter_h, filter_w, variance = var, filter_h_max = filter_h_max, filter_w_max = filter_w_max)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
        
def Notch_Reject_Filter():  
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x400")
    e = []
    res = []
    tk.Label(son, text= "类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "x方向频率：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "y方向频率：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "截止频率：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    tk.Label(son, text= "阶数：").pack()
    e5 = tk.Entry(son)
    e.append(e5)
    e5.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    u = float(res[1])
    v = float(res[2])
    cut_off = float(res[3])
    order = 0
    if len(res[4]) != 0:
        order = int(res[4])
    img = _Notch_Reject_Filter(img, _type, u, v, cut_off, order = order)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Notch_Pass_Filter():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x400")
    e = []
    res = []
    tk.Label(son, text= "类型：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "x方向频率：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "y方向频率：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "截止频率：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    tk.Label(son, text= "阶数：").pack()
    e5 = tk.Entry(son)
    e.append(e5)
    e5.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    _type = res[0]
    u = float(res[1])
    v = float(res[2])
    order = 0
    cut_off = float(res[3])
    if len(res[4]) != 0:
        order = int(res[4])
    img = _Notch_Pass_Filter(img, _type, u, v, cut_off, order = order)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Optimum_Notch_Filter():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x400")
    e = []
    res = []
    tk.Label(son, text= "窗口高度：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "窗口宽度：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "x方向频率：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    tk.Label(son, text= "y方向频率：").pack()
    e4 = tk.Entry(son)
    e.append(e4)
    e4.pack()
    tk.Label(son, text= "截止频率：").pack()
    e5 = tk.Entry(son)
    e.append(e5)
    e5.pack()
    tk.Label(son, text= "阶数：").pack()
    e6 = tk.Entry(son)
    e.append(e6)
    e6.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    filter_h = int(res[0])
    filter_w = int(res[1])
    u = float(res[2])
    v = float(res[3])
    cut_off = 5
    if len(res[4]) != 0:
        cut_off = float(res[4])
    order = 0
    if len(res[5]) != 0:
        order = int(res[5])
    img = _Optimum_Notch_Filter(img, filter_h, filter_w, u, v, cut_off = cut_off, order = order)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Motion_Degradation():
    global img
    fig1.clear()
    son = tkinter.Toplevel()
    son.title("输入参数")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "x方向速度：").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    tk.Label(son, text= "y方向速度：").pack()
    e2 = tk.Entry(son)
    e.append(e2)
    e2.pack()
    tk.Label(son, text= "T：").pack()
    e3 = tk.Entry(son)
    e.append(e3)
    e3.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    a = float(res[0])
    b = float(res[1])
    T= float(res[2])
    img = _Motion_Degradation(img, a, b, T)
    fig1.imshow(img, cmap = plt.get_cmap('gray'))
    canvas.draw()
    canvas.get_tk_widget().place(x = 300, y = 150)
    
def Haar_Fast_Wavelet():
    son = tkinter.Toplevel()
    son.title("输入序列")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "序列").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    seq = res[0].split(',')
    for i in range(len(seq)):
        seq[i] = float(seq[i])
    res = _Haar_Fast_Wavelet(seq)
    out = ''
    for num in res:
        out += str(num)
        out += ' '
    tkm.showinfo("结果", out)
    
def Reverse_Haar_Fast_Wavelet():
    son = tkinter.Toplevel()
    son.title("输入序列")
    son.geometry("200x200")
    e = []
    res = []
    tk.Label(son, text= "序列").pack()
    e1 = tk.Entry(son)
    e.append(e1)
    e1.pack()
    button1 = tkinter.Button(son, text = "确定", command = lambda: _Get_Argument(res, e, son))
    button1.pack()
    son.mainloop()
    seq = res[0].split(',')
    for i in range(len(seq)):
        seq[i] = float(seq[i])
    res = _Reverse_Haar_Fast_Wavelet(seq)
    out = ''
    for num in res:
        out += str(num)
        out += ' '
    tkm.showinfo("结果", out)
    
def Save_Img(img):
    f.savefig('local.png')
    
button1 = tkinter.Button(win, text = "选择图像", command = Choose_File)
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "直方图均衡", command = lambda: Histogram_Equalization())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "直方图匹配", command = lambda: Histogram_Matching())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "卷积", command = lambda: Conv())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "快速傅里叶变换", command = lambda: FFT())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "快速傅里叶反变换", command = lambda: Reverse_FFT())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "高通滤波器", command = lambda: High_Pass_Filter())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "低通滤波器", command = lambda: Low_Pass_Filter())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "随机噪声", command = lambda: Random_Noise())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "周期噪声", command = lambda: Periodic_Noise())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "空域滤波器", command = lambda: Filter_in_Space())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "空域适应性滤波器", command = lambda: Adaptive_Filter_in_Space())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "陷通滤波器", command = lambda: Notch_Pass_Filter())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "陷阻滤波器", command = lambda: Notch_Reject_Filter())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "最佳陷波滤波器", command = lambda: Optimum_Notch_Filter())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "运动模糊", command = lambda: Motion_Degradation())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "快速小波变换", command = lambda: Haar_Fast_Wavelet())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "快速小波反变换", command = lambda: Reverse_Haar_Fast_Wavelet())
button1.pack(side='top', anchor='nw')

button1 = tkinter.Button(win, text = "保存", command = lambda: Save_Img(img))
button1.pack(side='top', anchor='nw')

win.mainloop()
