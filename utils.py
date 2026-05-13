# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import h5py
import math
import glob
import os


def prepare_data(config):
    """
    返回图片文件夹和图片列表
    """
    if config.is_train:
        # 训练集文件夹路径
        input_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_input))
        input_list = glob.glob(os.path.join(input_dir, "*png"))
        label_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_label))
        label_list = glob.glob(os.path.join(label_dir, "*png"))

        # 验证集文件夹路径
        eval_input_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_input))
        eval_input_list = glob.glob(os.path.join(eval_input_dir, "*png"))
        eval_label_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_label))
        eval_label_list = glob.glob(os.path.join(eval_label_dir, "*png"))

        return input_list, label_list, eval_input_list, eval_label_list

    else:
        # 否则就是测试集
        test_dir = os.path.join(os.getcwd(), config.test_set)
        test_list = glob.glob(os.path.join(test_dir, "*.png"))

        return test_list


def input_setup(config):
    """
    调用相关函数制作train和eval的h5文件
    """
    input_list, label_list, eval_input_list, eval_label_list = prepare_data(config)
    print('Prepare training data...')
    make_sub_data(input_list, label_list, config, 'train')
    print('Prepare evaluating data...')
    make_sub_data(eval_input_list, eval_label_list, config, 'eval')


def make_data_hf(input_, label_, config, str, times):
    """
    制作h5文件
    """
    # assert input_.shape==label_.shape
    if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), "checkpoint"))
    if str == 'train':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5')
    elif str == 'eval':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'eval.h5')

    else:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:  # 在times=0的时候创建文件
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            # 写文件 h5py.File(file_name,'w')
            hf = h5py.File(savepath, 'w')
            if config.is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, config.c_dim),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim),
                                             dtype='float32')

                label_h5 = hf.create_dataset("label", (1, config.image_size, config.image_size, config.c_dim2),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim2),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim2),
                                             dtype='float32')
                a = 1

            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             dtype='float32')
    else:  # times!=0的时候,追加"a"
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]

    # 在这里增加维度,同时在这里将数据添加进来
    if config.is_train:
        input_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim2])
        label_h5[times: times + 1] = label_
    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times: times + 1] = label_

    hf.close()
    return True


def make_sub_data(input_list, label_list, config, str):
    """
    将图片拆分成输入大小,并存入h5文件
    :param input_list: input图片
    :param label_list: label图片
    :param config: 标签
    :param str: train或者eval
    :return: 图片列表
    """

    assert len(input_list) == len(label_list)
    times = 0  # 统计最后一张图变成多少张图
    for i in range(len(input_list)):
        input_1 = cv2.imread(input_list[i], -1)
        input_ = 0.6 * input_1 / 255.0

        label_1 = cv2.imread(label_list[i], -1)
        label_1 = 0.6 * label_1 / 255.0
        label_ = label_1[:, :, 0:1] + label_1[:, :, 2:3]
        label_ = label_ / np.nanmax(label_)
        #
        # cv2.namedWindow("imgS0")
        # # cv2.imshow("imgS0", label_)
        # cv2.imshow("imgS0", input_)
        # cv2.waitKey(0)
        #
        # print(input_.shape)
        # print(label_.shape)
        # assert input_.shape == label_.shape
        # 输入图片的维度是3:(1024, 1224, 4)
        # 计算Dolp等
        # path =
        # input_s0,input_dolp = cal_stokes_dolp(input_)
        # label_s0,label_dolp = cal_stokes_dolp(label_)
        # imsave_s0_dolp(input_s0,input_dolp,input_list[i],)
        # imsave_s0_dolp(label_s0,label_dolp,input_list[i],)
        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape

        # 如果不是在训练阶段，就直接将其归一化然后返回即可，不需要进行h5文件的制作
        if not config.is_train:
            make_data_hf(input_, label_, config, times)
            return input_list, label_list
        # h_stride = h // 8  # 128
        # w_stride = w // 8  # 153
        # 我这样制作的数据集产生重叠,步长是大小的一半,这里image_size =6 4,stride = 32
        # 最后的结果就是 (1024-32)/32 与 (1216-32)/32 所以最后一张图变成1147张
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                sub_label = label_[x:x + config.image_size, y:y + config.image_size]
                save_flag = make_data_hf(sub_input, sub_label, config, str, times)
                if not save_flag:
                    return input_list, label_list
                times += 1
        print("image: [%2d], total: [%2d]" % (i, len(input_list)))
    return input_list, label_list


# 获取训练集或者验证集图片数量
def get_data_num(path):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]


# 获取数据h5地址
def get_data_dir(checkpoint_dir, is_train):
    if is_train:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5'), os.path.join(
            os.path.join(os.getcwd(), "checkpoint"), 'eval.h5')

    else:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'test.h5')


# 随机获取一个batch的数据
def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r')as hf:
        # print(hf.shape)shape
        input_ = hf["input"]
        label_ = hf["label"]
        # 返回一个有batch_size个分布在0~1之间元素的数组,数组里面的数服从正态分布,然后再乘以训练集图片个数,那么产生了(0-图像总个数-1)的随机数了
        random_batch = np.random.rand(batch_size) * (data_num - 1)
        # 最后batch_images.shape = (64,64,64,4)
        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

        random_aug = np.random.rand(2)  # 产生两个元素的数组,这两个元素决定翻转与否
        # 只要将该随机数保存一致,然后作为训练数据和测试数据的增强参数就可以保证对应起来了
        batch_images = augmentation(batch_images, random_aug)
        batch_labels = augmentation(batch_labels, random_aug)
        return batch_images, batch_labels


def augmentation(batch, random):
    if random[0] < 0.3:
        # 在batch的第shape[1]上，上下翻转
        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:
        # 在batch的第shape[2]上，上下翻转
        batch_flip = np.flip(batch, 2)
    else:
        # 不翻转
        batch_flip = batch

    if random[1] < 0.5:
        # 在翻转的基础上旋转
        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:
        # 不翻转
        batch_rot = batch_flip

    return batch_rot


# 显示图片,调bug用
def show_img(img, k):
    if k == 'm':
        img = img
        cv2.imshow("img_merge", img)
        cv2.waitKey(0)
    if k == '0':
        img = img[:, :, 0]
        cv2.imshow("img_0", img)
        cv2.waitKey(0)
    if k == '45':
        img = img[:, :, 1]
        cv2.imshow("img_45", img)
        cv2.waitKey(0)
    if k == '90':
        img = img[:, :, 2]
        cv2.imshow("img_90", img)
        cv2.waitKey(0)
    if k == '135':
        img = img[:, :, 3]
        cv2.imshow("img_135", img)
        cv2.waitKey(0)


def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def save_image(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    path_m = os.path.join(config.output_dir, "merge")
    if not os.path.isdir(path_m):
        os.mkdir(path_m)
    path_e = os.path.join(config.output_dir, "each_channel")
    if not os.path.isdir(path_e):
        os.mkdir(path_e)
    path_f = os.path.join(config.output_dir, "full_size")
    if not os.path.isdir(path_f):
        os.mkdir(path_f)
    imsave(img, path_m + '/%s-m.png' % output_name)
    imsave(img0, path_e + '/%s-0.png' % output_name)
    imsave(img45, path_e + '/%s-45.png' % output_name)
    imsave(img90, path_e + '/%s-90.png' % output_name)
    imsave(img135, path_e + '/%s-135.png' % output_name)
    imsave(dofp, path_f + '/%s-f.png' % output_name)


def save_image0(config, img, output_name):
    img0 = img[:, :, 0]
    img90 = img[:, :, 1]
    path_e0 = os.path.join(config.output_dir, "0_channel")
    if not os.path.isdir(path_e0):
        os.mkdir(path_e0)
    imsave(img0, path_e0 + '/%s-0.png' % output_name)
    path_e90 = os.path.join(config.output_dir, "90_channel")
    if not os.path.isdir(path_e90):
        os.mkdir(path_e90)
    imsave(img90, path_e90 + '/%s-90.png' % output_name)


def save_imageS0(config, img, output_name):
    imgS0 = img * 255
    path_s0 = os.path.join(config.output_dir, "s0_channel")
    if not os.path.isdir(path_s0):
        os.mkdir(path_s0)
    imsave(imgS0, path_s0 + '/%s-s0.png' % output_name)


def psnr(img1, img2):
    return tf.reduce_mean(tf.image.psnr(img1, img2, 1.0))  # 如果图片没有归一化改成255
