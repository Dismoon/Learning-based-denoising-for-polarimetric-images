import cv2
import os
import glob
import numpy as np
# import matplotlib


def RGB2Gray(path,data_num):
    path_g = os.path.join(path, "灰度图")
    if not os.path.isdir(path_g):
        os.mkdir(path_g)
    for i in range(1, data_num + 1):
        img = cv2.imread(path + '/' + "%s.png" % i, 2)
        cv2.imwrite(path_g + '/' + "%s.png" % i, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0




def Downsample(path, data_num, scale):
    path_d = os.path.join(path, "%d倍降采样" % scale)
    if not os.path.isdir(path_d):
        os.mkdir(path_d)
    for i in range(1, data_num+1):
        input_path = path + "/%d-m.png" % i
        input = cv2.imread(input_path, -1)
        # input = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)  # 转灰度
        h, w,_ = input.shape
        # img_new = cv2.resize(input, (int(w*3), int(h*3)), interpolation=cv2.INTER_CUBIC)
        img_new = cv2.resize(input, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_LINEAR) #　降采样
        cv2.imwrite(os.path.join(path_d, '%d-%d.png' % (i, scale)), img_new, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def get_avarage(path,star_num,end_num,n):
    path_a = os.path.join(path, "avarage")
    if not os.path.isdir(path_a):
        os.mkdir(path_a)
    for i in range(star_num, end_num+1):
        img = cv2.imread(path + '/' + "%s-l-0.png" % i, cv2.IMREAD_GRAYSCALE)
        sum = np.zeros(img.shape)
        for j in range(0,n):
            img = cv2.imread(path + '/' + "%s-l-%s.png" % (i, j), cv2.IMREAD_GRAYSCALE)
            sum = sum + img
        average_img = sum / 10.0

        cv2.imwrite(path_a + '/' + "%s-l.png" % i, average_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def merge(path,star_num,end_num):
    path_m = os.path.join(path, "merge")
    if not os.path.isdir(path_m):
        os.mkdir(path_m)
    for i in range(star_num, end_num+1):
        img = cv2.imread(path + '/' + "%s-s-0.png" % i, cv2.IMREAD_GRAYSCALE)

        # 索引顺序：(行，列),也就是(纵向(y),横向(x))
        img0 = img[1024:2049, 1224:2449]
        img45 = img[0:1024, 1224:2449]
        img90 = img[0:1024, 0:1224]
        img135 = img[1024:2049, 0:1224]

        merged = cv2.merge([img0, img45, img90, img135])
        cv2.imwrite(path_m + '/' + "%s-m.png" % i, merged, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def copy(path,star_num,end_num):
    path_m = os.path.join(path, "merge")
    if not os.path.isdir(path_m):
        os.mkdir(path_m)
    for i in range(star_num, end_num+1):
        img = cv2.imread(path + '/' + "%s_0_000.png" % i, -1)

        # 索引顺序：(行，列),也就是(纵向(y),横向(x))
        img0 = img
        img45 = img
        img90 = img

        merged = cv2.merge([img0, img45,img90])
        cv2.imwrite(path_m + '/' + "%s-m.png" % i, merged, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def merge1(path,path1,data_num):
    path_m = os.path.join(path, "merge")
    if not os.path.isdir(path_m):
        os.mkdir(path_m)
    for i in range(1, data_num+1):
        path_0 = path + "/" + "%d_4_000.png" % i
        # path_45 = path + "/" + "%d_5_045.png" % i
        path_90 = path1 + "/" + "%d_4_090.png" % i
        # path_135 = path + "/" + "%d_5_135.png" % i

        img0 = cv2.imread(path_0, cv2.IMREAD_GRAYSCALE)
        # img45 = cv2.imread(path_45, cv2.IMREAD_GRAYSCALE)
        img90 = cv2.imread(path_90, cv2.IMREAD_GRAYSCALE)
        # img135 = cv2.imread(path_135, cv2.IMREAD_GRAYSCALE)

        merged = cv2.merge([img0,img0,img0,img0,img90])
        cv2.imwrite(path_m + '/' + "%s-2-m.png" % i, merged, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def get_dolp_aop(path,star_num,end_num):
    path_d = os.path.join(path, "dolp")
    if not os.path.isdir(path_d):
        os.mkdir(path_d)
    path_a = os.path.join(path, "aop")
    if not os.path.isdir(path_a):
        os.mkdir(path_a)
    for i in range(star_num, end_num + 1):
        img = cv2.imread(path + '/' + "%d-m.png" % i, -1)
        i0 = img[:,:,0]
        i0 = i0.astype(float)
        i45 = img[:,:,1]
        i45 = i45.astype(float)
        i90 = img[:,:,2]
        i90 = i90.astype(float)
        i135 = img[:,:,3]
        i135 = i135.astype(float)

        s0 = (i0 + i45 + i90 + i135) / 2.0
        s1 = i0 - i90
        s2 = i45 - i135
        aop = 1 / 2 * np.arctan2(s2, s1)
        aop = aop * (1 / np.nanmax(aop))  # 归一化到1内
        aop = np.clip(aop * 255, 0, 255)  # 然后归一化到0-255
        dolp = 255 * (np.power(np.power(s1,2) + np.power(s2,2),0.5) / (s0 + 0.00001))

        cv2.imwrite(path_a + '/' + "%s-a.png" % i, aop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(path_d + '/' + "%s-d.png" % i, dolp, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def make_sub_data(path,num,size):
    path_d = os.path.join(path, "sub")
    if not os.path.isdir(path_d):
        os.mkdir(path_d)
    for i in range(1, num + 1):
        img = cv2.imread(path + "/%d-6.png" % i, -1)
        h, w= img.shape
        for x in range(0, h-size+1,size):
            for y in range(0, w-size+1,size):
                sub_input = img[x:x+size,y:y+size]
                # sub_input = sub_input / 255.0  # 归一化
                cv2.imwrite(path_d + '/' + "%s-%s-%s.png" %(i,x,y), sub_input, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    return 0

def Upsample(path,data_num,scale):
    path_u = os.path.join(path, "三次样条恢复")
    if not os.path.isdir(path_u):
        os.mkdir(path_u)
    for i in range(1, data_num+1):
        input_path = path + "/%d-%d.png" % (i,scale)
        input = cv2.imread(input_path,-1)
        # input = cv2.imread(input, cv2.IMREAD_GRAYSCALE)  # 转灰度
        h, w,_ = input.shape
        img_new = cv2.resize(input, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)  # 恢复
        cv2.imwrite(os.path.join(path_u, '%dx%d.png' % (i, scale)), img_new, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


def cut(path,star_num,end_num):
    path_c = os.path.join(path, "cut")
    if not os.path.isdir(path_c):
        os.mkdir(path_c)
    for i in range(star_num, end_num+1):
        img = cv2.imread(path + '/' + "%s.png" % i, -1)

        # 索引顺序：(行，列),也就是(纵向(y),横向(x))
        h,w = img.shape
        img = img[20:h-20, 20:w-20]

        cv2.imwrite(path_c + '/' + "%s.png" % i, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return 0


# make_sub_data("/home/polarization/Nutstore Files/see in the dark/超分重建/线/灰度图/6倍降采样",63,20)
# get_avarage(r"D:\see in the dark\data\dataset10-5",60,150,10)
# copy("/media/polarization/My Passport/zhangyanbin/实验数据/深度学习/数据2/backups_aver/0_000",1,84)
merge1("/media/polarization/My Passport/zhangyanbin/实验数据/深度学习/数据2/backups_aver/4_000","/media/polarization/My Passport/zhangyanbin/实验数据/深度学习/数据2/backups_aver/4_090",70)
# get_dolp_aop("/home/polarization/Nutstore Files/see in the dark/data/data_all/label",1,126)
# cut("/home/polarization/Nutstore Files/see in the dark/超分重建/线/灰度图",1,36)
# RGB2Gray("/home/polarization/Nutstore Files/see in the dark/超分重建/线/原图",84)
# Downsample("/home/polarization/Nutstore Files/see in the dark/超分重建/线/灰度图/merge",84,4)
# Upsample("/home/polarization/Nutstore Files/see in the dark/超分重建/线/灰度图/merge/4倍降采样",36,4)
