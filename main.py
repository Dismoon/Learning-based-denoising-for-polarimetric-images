# -*- coding: utf-8 -*-
import cv2
import glob
import os
from utils import *

import tensorflow as tf
from model import RDN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")  # 是否训练
flags.DEFINE_boolean("is_eval", True, "if the evaluation")  # 是否验证
flags.DEFINE_string("train_set_input", "train/input", "name of the train input set")  # 训练集input文件夹
flags.DEFINE_string("train_set_label", "train/label", "name of the train label set")  # 训练集label文件夹
flags.DEFINE_string("eval_set_input", 'eval/input', 'eval_set_input')  # 验证集的input文件夹
flags.DEFINE_string("eval_set_label", 'eval/label', 'eval_set_label')  # 验证集的label文件夹
flags.DEFINE_string("output_dir", "test/output", "test output")  # 测试集output文件夹
flags.DEFINE_string("test_set", "test/input", "test input")  # 测试集input文件夹
flags.DEFINE_integer("image_size", 64, "the height of image input")  # 输入网络图片大小为64*64
flags.DEFINE_integer("c_dim", 3, "the size of channel")  # 图片的通道是4个通道的
flags.DEFINE_integer("c_dim2", 1, "the size of channel")  # 输出图片的通道是4个通道的
flags.DEFINE_integer("stride", 64, "the size of stride")  # 切割图片的步长
flags.DEFINE_integer("epoch", 100, "number of epoch")  # epoch
flags.DEFINE_integer("batch_size", 32, "the size of batch")  # batch size
flags.DEFINE_float("learning_rate", 1e-4, "the learning rate")  # learning rate
flags.DEFINE_float("lr_decay_steps", 10, "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.6, "rate of learning rate decay")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")  # checkpoint文件夹
flags.DEFINE_integer("D", 16, "D")  # Residual Dense Block块的个数
flags.DEFINE_integer("C", 6, "C")  # 每个Residual Dense Block块中conv层数量
flags.DEFINE_integer("G", 16, "G")  # 模型中所有层输出的feature maps不是G就是G0
flags.DEFINE_integer("G0", 32, "G0")  # 在DRB中的ReLU前面都由G0个特征提取器
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")  # 卷积核的大小


def main(_):
    rdn = RDN(tf.Session(),
              is_train=FLAGS.is_train,
              is_eval=FLAGS.is_eval,
              image_size=FLAGS.image_size,
              c_dim=FLAGS.c_dim,
              c_dim2=FLAGS.c_dim2,
              batch_size=FLAGS.batch_size,
              D=FLAGS.D,
              C=FLAGS.C,
              G=FLAGS.G,
              G0=FLAGS.G0,
              kernel_size=FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)


if __name__ == '__main__':
    # 执行程序中main函数，并解析命令行参数！
    tf.app.run()
