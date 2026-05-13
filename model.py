# -*- coding: utf-8 -*-
"""
去上采样，去残差
"""
import tensorflow as tf
import numpy as np
import time
import os
from utils import *


# LEARNING_RATE_BASE = 0.001
# LEARNING_RATE_DECAY = 0.9
class RDN(object):

    def __init__(self,
                 sess,  # 用来传递一个TensorFlow会话
                 is_train,
                 is_eval,  # is_train和is_eval用来控制训练还是测试
                 image_size,  # img_size是输入图片大小
                 c_dim,  # 图片通道数
                 c_dim2,  # 输出图片通道数
                 batch_size,
                 D,  # Residual Dense Block块的个数
                 C,  # 每个Residual Dense Block块中conv层数量
                 G,
                 G0,  # 模型中所有层输出的feature maps不是G就是G0
                 kernel_size  # 卷积核的大小
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.c_dim2 = c_dim2
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size

    def SFEParams(self):
        """
                浅层特征提取部分（两个conv层，产生F_-1和F_0）
                最后输出有G个feature maps

        	    卷积核是一个四维的tensor -->（ks, ks, self.c_dim, G0）
        	    前两个参数是卷积核kernel的size
        	    第三个是输入tensor的通道数
        	    第四个是输出tensor的通道数
        	    偏置单元和输出通道数保持一致
        """
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': tf.get_variable(name='w_S_1',shape=[ks, ks, self.c_dim, G0], initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w_S_2': tf.get_variable(name='w_S_2',shape=[ks, ks, G0, G], initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w_S_3': tf.get_variable( name='w_S_3',shape=[ks, ks, G, G], initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w_S_4': tf.get_variable(name='w_S_4',shape=[ks, ks, G, G], initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w_S_5': tf.get_variable( name='w_S_5',shape=[ks, ks, G, G], initializer=tf.contrib.layers.variance_scaling_initializer())
        }
        biasesS = {
            'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
            'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2')),
            'b_S_3': tf.Variable(tf.zeros([G], name='b_S_3')),
            'b_S_4': tf.Variable(tf.zeros([G], name='b_S_4')),
            'b_S_5': tf.Variable(tf.zeros([G], name='b_S_5'))
        }

        return weightsS, biasesS

    def RDBParams(self):
        """
        	    第i个RDB块接受第i-1个RDB块传来的输出作为输入，在每个RDB块中，每一层的输出都会送个它的后面所有层。第D个RDB块的第c层输出的公式如下：
        	    F_{d,c}=\sigma(W_{d,c}[F_{d-1},F_{d,1},F_{d,2}…F_{d,c-1}])

        	    其中[Fd−1,Fd,1,Fd,2...Fd,c−1]就是将它们concat在一起，也即包含[G0+(c−1)∗G]个feature maps。

        	    每个RDB块由以下模块装成（conv1 -> relu1 -> conv2 -> relu2 … -> convC ->reluC -> concatnation -> 1*1 conv -> local residual）
        """
        weightsR = {}
        biasesR = {}
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size

        for i in range(1, D + 1):
            # 第i个稠密块的稠密卷积 dense conv layers in i-th dense block
            for j in range(1, C + 1):
                weightsR.update({'w_R_%d_%d' % (i, j): tf.get_variable(name='w_R_%d_%d' % (i, j),shape=[ks, ks, G * j, G], initializer=tf.contrib.layers.variance_scaling_initializer())})
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
                # 第i个稠密块的局部特征融合 local feature fusion in i-th dense block
            weightsR.update({'w_R_%d_%d' % (i, C + 1): tf.get_variable(name='w_R_%d_%d' % (i, C + 1),shape=[1, 1, G * (C + 1), G], initializer=tf.contrib.layers.variance_scaling_initializer())})
            biasesR.update({'b_R_%d_%d' % (i, C + 1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C + 1)))})

        return weightsR, biasesR

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D + 1):
            x = rdb_in
            for j in range(1, self.C + 1):
                tmp = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, j)], strides=[1, 1, 1, 1], padding='SAME') + \
                      self.biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                # 在最后一个维度做concat操作
                x = tf.concat([x, tmp], axis=3)

            # local feature fusion
            x = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, self.C + 1)], strides=[1, 1, 1, 1], padding='SAME') + \
                self.biasesR['b_R_%d_%d' % (i, self.C + 1)]
            # local residual learning
            rdb_in = tf.add(x, rdb_in)
            # 为global feature fusion做准备
            rdb_concat.append(rdb_in)
        # 在最后一个维度做concat
        return tf.concat(rdb_concat, axis=3)

    def DFFParams(self):
        """
	    这一部分主要是将前面所有RDB的结果进行一个特征融合，方法和RDB块中最后的concat操作类似，参阅模型整体图的三个红色块后面的concat操作，然后对concated tensor做1∗1卷积到G个feature maps，再进行 3*3 卷积准备进行Global residual learning。公式如下：FGF=HGFF([F1,F2...FD])
	    """
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsD = {
            'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=0.01), name='w_D_1'),
            'w_D_2': tf.get_variable(name='w_D_2', shape=[ks, ks, G0, G0], initializer=tf.contrib.layers.variance_scaling_initializer())
        }
        biasesD = {
            'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
            'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2'))
        }

        return weightsD, biasesD

    def model(self):
        # SFE部分
        F_1 = tf.nn.conv2d(self.images, self.weightsS['w_S_1'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesS[
            'b_S_1']
        F0 = tf.nn.conv2d(F_1, self.weightsS['w_S_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesS['b_S_2']
        # F1 = tf.nn.conv2d(F0, self.weightsS['w_S_3'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesS['b_S_3']
        # F2 = tf.nn.conv2d(F1, self.weightsS['w_S_4'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesS['b_S_4']
        # F3 = tf.nn.conv2d(F2, self.weightsS['w_S_5'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesS['b_S_5']

        # RDBs部分
        FD = self.RDBs(F0)

        # DFF部分，1*1卷积再3*3卷积
        FGF1 = tf.nn.conv2d(FD, self.weightsD['w_D_1'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesD['b_D_1']
        FGF2 = tf.nn.conv2d(FGF1, self.weightsD['w_D_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biasesD['b_D_2']

        # Global Residual Learning部分
        # FDF = tf.add(FGF2, F_1)
        IHR = tf.nn.conv2d(FGF2, self.weight_final, strides=[1, 1, 1, 1], padding='SAME') + self.bias_final
        PSNR = psnr(IHR, self.labels)

        return IHR, PSNR

    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        # label是ground truth
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')
        self.weightsS, self.biasesS = self.SFEParams()
        self.weightsR, self.biasesR = self.RDBParams()
        self.weightsD, self.biasesD = self.DFFParams()

        # 最后一个conv层
        self.weight_final = tf.Variable(
            tf.random_normal([self.kernel_size, self.kernel_size, self.G0, self.c_dim2], stddev=np.sqrt(2.0 / 9 / 3)),
            name='w_f')
        self.bias_final = tf.Variable(tf.zeros([self.c_dim2], name='b_f'))

        self.pred,self.psnr = self.model()
        # 损失函数,square求平方:L2误差 abs求绝对值:L1误差
        # self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.pred))

        #tensorborad 标签
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('PSNR', self.psnr)

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self, config):

        input_setup(config)  # 这里创建了数据集的h5文件

        # 在这里获取训练集和验证集的h5文件
        train_data_dir, eval_data_dir = get_data_dir(config.checkpoint_dir, config.is_train)
        # 这里获取的是训练集图片总数,然后指数衰减学习率中的设置为data_num//self.batch_size,表示每个epoch衰减一次
        train_data_num = get_data_num(train_data_dir)
        # 训练一轮要多少个batch
        batch_num = train_data_num // config.batch_size
        # 这里获取验证集的data_num
        eval_data_num = get_data_num(eval_data_dir)
        # print("train_data_num",train_data_num)
        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim2]
        self.build_model(images_shape, labels_shape)

        # train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        # tf.global_variables_initializer().run(session=self.sess)

        epoch, counter = self.load(config.checkpoint_dir)
        global_step = tf.Variable(counter, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps * batch_num,
                                                   config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_train_path = os.path.join(config.checkpoint_dir, "train_%s_%s_%s" % (self.D, self.C, self.G))
        summary_eval_path = os.path.join(config.checkpoint_dir, "eval_%s_%s_%s" % (self.D, self.C, self.G))

        summary_writer_train = tf.summary.FileWriter(summary_train_path, self.sess.graph)
        summary_writer_validate = tf.summary.FileWriter(summary_eval_path)

        # 保存计算图到文件（用于tensorboard可视化）
        # 继续模型之前的计算
        time_all = time.time()
        print("\nNow Start Training...\n")

        # 断点续传
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))

        for ep in range(epoch, config.epoch):
            # Run by batch images

            for idx in range(0, batch_num):
                batch_images, batch_labels = get_batch(train_data_dir, train_data_num, config.batch_size)
                # 在这里增加了验证数据
                eval_batch_images, eval_batch_labels = get_batch(eval_data_dir, eval_data_num, config.batch_size)
                counter += 1
                # assert batch_images.shape == batch_labels.shape
                # assert eval_batch_images.shape == eval_batch_labels.shape
                _, loss, lr, psnr = self.sess.run([learning_step, self.loss, learning_rate, self.psnr],
                                           feed_dict={self.images: batch_images, self.labels: batch_labels})
                # 在这里增加了验证代码
                eval_loss = self.sess.run(self.loss,
                                          feed_dict={self.images: eval_batch_images,
                                                     self.labels: eval_batch_labels})
                if counter % 10 == 0:
                    print(
                                "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min, psnr:[%2.4f], train_loss: [%.8f],eval_loss:[%.8f]" % (
                            ep + 1, idx, batch_num, counter, int((time.time() - time_all)/60), psnr, loss, eval_loss))

                # 每500个step保存一次checkpoint
                if counter % 100 == 0:
                    print(int((time.time() - time_all)/60))
                    self.save(config.checkpoint_dir, ep + 1, counter)
                    summary_train = self.sess.run(merged_summary_op,
                                                  feed_dict={self.images: batch_images, self.labels: batch_labels})
                    summary_writer_train.add_summary(summary_train, counter)
                    # 写入evallogmessage
                    summary_eval = self.sess.run(merged_summary_op,
                                                 feed_dict={self.images: eval_batch_images,
                                                            self.labels: eval_batch_labels})
                    summary_writer_validate.add_summary(summary_eval, counter)
                if counter > 0 and counter == batch_num * config.epoch:
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    return

            # 每个epoch都验证一下

    def test(self, config):
        print("\nPrepare Testing Data...\n")
        paths = prepare_data(config)  # 最后返回的是路径列表
        data_num = len(paths)

        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            output_name = paths[idx].split("/")[-1].split('.')[0]
            # input_ = imread(paths[idx]) # 我们只需要读取
            input_1 = cv2.imread(paths[idx], -1)
            # input_ = 0.6 * (input_1[:, :, 0:1] + input_1[:, :, 2:3])/255.0
            input_1 = 0.6 * input_1 / 255.0
            label2_ = (input_1[:, :, 0:1] + input_1[:, :, 2:3])
            # input_2=input_1[:,:,0:3:2]
            input_ = input_1[np.newaxis, :]
            # label2_ = label2_[:, np.newaxis]# 输入的维度是(batch,h,w,channels)所以需要增加一个维度

            # np.sarray将数据转换成ndarray
            images_shape = input_.shape
            # label2_ = 0.6 * (input_[:, :, 0:1] + input_[:, :, 2:3])/255.0

            labels_shape = label2_.shape
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess)

            self.load(config.checkpoint_dir)
            result = self.sess.run([self.pred], feed_dict={self.images: input_})
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            # 因为result输出的维度是(1,h,w,4),丫说维度变成(h,w,4)
            img = np.squeeze(result)
            img = 1 / np.nanmax(img) * img
            save_imageS0(config, img, output_name)
            print('Running')
        print("\n All Done ! ")

    def load(self, checkpoint_dir):
        """
                关于tf.train.get_checkpoint_state(checkpoint_dir,latest_filename=None)：
        		返回：checkpoint文件CheckpointState proto类型的内容，
        		其中有model_checkpoint_path和all_model_checkpoint_paths两个属性。

        		model_checkpoint_path：保存了最新的tensorflow模型文件的文件名，
        		all_model_checkpoint_paths：则有未被删除的所有tensorflow模型文件的文件名。
        """
        # 从指定目录加载模型已经计算的部分，并接着计算
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            print(os.path.join(os.getcwd(), ckpt_path))
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(ckpt_path.split('-')[-1])
            epoch = int(ckpt_path.split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            epoch = 0
            print("\nCheckpoint Loading Failed! \n")

        return epoch, step

    def save(self, checkpoint_dir, epoch, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_name = os.path.join(checkpoint_dir, model_name + '-{}'.format(epoch + 1))
        self.saver.save(self.sess,
                        save_name,
                        global_step=step)

