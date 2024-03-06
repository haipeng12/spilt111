import os
import cv2
from torch.utils.data import Dataset
class CUB(Dataset):

    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        with open(os.path.join(self.root, 'order_imagename.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'order_label.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        self.data_id = []
        if self.is_train:#训练集
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, train = line.split()
                    if train=='1':
                        self.data_id.append(image_id)
        if not self.is_train:#测试集
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, train = line.split()
                    if train=='0':
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        #index可能是len（data_id）包括0
        image_id = self.data_id[index]                              #输入一个序号，并获得train_test_split里对应的order
        class_id = int(self._get_class_by_id(image_id)) - 1         #根据order，在order-label找到对应类别
        path = self._get_path_by_id(image_id)                       #根据order，在order-imagename找到对应图像路径
        image = cv2.imread(os.path.join(self.root, 'image', path)) #根据图片路径获取对应图片
        #print(image_id)
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
