'''
    TFRecode I/O 流控制器
            2018-06-14
            by-fffeiyee
'''
import os
import tensorflow as tf
from tensorflow.contrib import data
from configobj import ConfigObj
from tqdm import tqdm

# IN
class TFRecodeLib():
    def __init__(self):
        self.config = ConfigObj("Unit.cfg")
        section = self.config['Image data']
        self.train_cfg = self.config['Train data']
        self.test_cfg = self.config['Test data']
        self.file_root = section['data_root']
        self.data_path = section['image_path']
        self.instances_per_shard = 500  # 定义每个文件的写入数量
        self.train_ratio = float(section['train_ratio'])
        self.train_file,self.test_file = self.get_file_name(self.file_root + self.data_path)
        self.file_name = []
        self.image_w = int(section['image_w'])
        self.image_h = int(section['image_h'])
        print("Init is success")



    def get_file_name(self,file_dir):
        '''
        获取路径下所有的文件路径
        :param file_dir: 源路径
        :return: 训练List ， 测试List
        '''
        i = 0
        temp_dir = []
        train_path = []
        test_path = []
        for _, dirs, files in os.walk(file_dir):
            if len(dirs) > 0:
                temp_dir = dirs
            if len(files) > 0:
                j = 0
                for file in files:
                    if j <= len(files) * self.train_ratio:
                        train_path.append([int(temp_dir[i]),temp_dir[i] + "/" + file])
                        j += 1
                    else:
                        test_path.append([int(temp_dir[i]),temp_dir[i] + "/" + file])
                i += 1

        return train_path,test_path

    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save_tfrecode(self,data,tfr_name):
        '''
        生成TFRecode
        :param data: 数据集
        :param tfr_name: 文件名
        :return: None
        '''
        num_shards = len(data) / self.instances_per_shard
        if num_shards > float(int(num_shards)):
            num_shards = int(num_shards) + 1
        else:
            num_shards = int(num_shards)
        image_index = 0
        for i in tqdm(range(num_shards),desc="封装总进度："):
            file_name = (self.file_root + "TFRecode/" + tfr_name + '.tfrecodes-%.2d-of-%.2d'%(i,num_shards))
            writer = tf.python_io.TFRecordWriter(file_name)
            for j in tqdm(range(self.instances_per_shard),desc="封装" + str(i) + " : "):
                if image_index == len(data):
                    break
                image_lable = data[image_index][0]
                image_file_name = data[image_index][1]
                image_raw_data = tf.gfile.FastGFile(self.file_root + self.data_path + image_file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                image = tf.image.resize_images(image,[self.image_w,self.image_h])
                image = tf.image.per_image_standardization(image)
                image = image.eval(session=tf.Session())
                image = image.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image':self._bytes_feature(image),
                    'lable':self._int64_feature(image_lable)
                }))
                writer.write(example.SerializeToString())
                image_index += 1
            writer.close()
            if image_index == len(data):
                break

    def save_all(self):
        '''
        自动生成TFRecode
        :return: None
        '''
        print("Train num :",len(self.train_file))
        print("Test num : ",len(self.test_file))
        self.train_cfg['num_train'] = len(self.train_file)
        self.test_cfg['num_test'] = len(self.test_file)
        self.config.write()
        self.save_tfrecode(self.train_file,"Train")
        self.save_tfrecode(self.test_file,"Test")
        print("Save is success")
# OUT
class DataSetLib():
    def __init__(self,sess,select_class,image_shape,batch_size):
        self.recode_path = "LiaoNing/TFRecode/"
        self.sess = sess
        self.select_class = select_class
        self.image_w = int(image_shape[0])
        self.image_h = int(image_shape[1])
        self.batch_size = batch_size
        self.shuffle = True
        self.train_recode , self.test_recode = self.get_recode_name()
        self.data = self.train_recode
        if self.select_class == "Test" or self.select_class == "test":
            self.data = self.test_recode
            self.shuffle = False
        elif self.select_class is not "Train" and self.select_class is not "train":
            raise RuntimeError('select is must "Test" or "Train" !')
        print("Init is success")
    def get_recode_name(self):
        '''
        获取文件夹下所有文件名
        :return: 训练文件List ， 测试文件List
        '''
        files = []
        train_recode = []
        test_recode = []
        for root, dirs, file in os.walk(self.recode_path):
            if len(file) > 0:
                files = file
                break
        for word in files:
            if "Train" in word or "train" in word:
                train_recode.append(self.recode_path + word)
            elif "Test" in word or "test" in word:
                test_recode.append(self.recode_path + word)
        return train_recode,test_recode
    # TFRecords 解析
    def parse(self,record):
        features = tf.parse_single_example(
            record,
            features={
                'image': tf.FixedLenFeature([], tf.string, default_value=None),
                'lable': tf.FixedLenFeature([], tf.int64, default_value=None)
            }
        )
        image_data = tf.decode_raw(features['image'],tf.float32)
        label = features['lable']
        return image_data,label

    def image_norm(self,image,shape):
        '''
        cifar-10 图像标准化
        :param image:   图像通道
        :param shape:   理想维度
        :return:        标准化图像通道
        '''
        image = tf.image.per_image_standardization(tf.reshape(image, shape))
        image = tf.reshape(image,[3,self.image_w,self.image_h,1])
        image = tf.concat([image[0], image[1], image[2]], 2)
        return image

    def total_image_norm(self,image,shape):
        '''
        通用图像标准化
        :param image:   图像通道
        :param shape:   理想维度
        :return:        标准化图像通道
        '''
        image = tf.reshape(image, shape)
        return image

    def get_batch_data(self):
        '''
        获取 Batch size 数据
        :return:  图像Tensor ， 标签Tensor
        '''
        dataSet = data.TFRecordDataset(self.data)
        dataSet = dataSet.map(self.parse)
        dataSet = dataSet.map(lambda image,label:(self.total_image_norm(image,[self.image_w,self.image_h,3]),label))
        dataSet = dataSet.repeat()
        if self.shuffle:
            dataSet = dataSet.shuffle(1000)
        dataSet = dataSet.batch(self.batch_size)
        iterator = dataSet.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(iterator.initializer)
        label_batch = tf.reshape(label_batch,[-1,1])
        return image_batch,label_batch

'''
生成TFRecode
'''
if __name__ == '__main__':
    a = TFRecodeLib()
    a.save_all()