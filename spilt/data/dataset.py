# -*- coding: utf-8 -*-

from data.dataloader import *
from data.transforms import *
from torch.utils.data import DataLoader
#from data.Weightsample import *
#from torch.utils.data.sampler import WeightedRandomSampler
IMAGE_SIZE = 224
TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

path = 'data'
train_transforms = Compose([
        ToCVImage(),
        RandomResizedCrop(IMAGE_SIZE),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

test_transforms = Compose([
    ToCVImage(),
    Resize(IMAGE_SIZE),
    ToTensor(),
    Normalize(TEST_MEAN,TEST_STD)
    ])
train_dataset = CUB(path,
        train=True,
        transform=train_transforms,
        target_transform=None
    )
    # print(len(train_dataset))

def returntrain_dataloader(batch_size):
    return DataLoader(train_dataset,batch_size=batch_size,num_workers=0,shuffle=True)

test_dataset = CUB(path,
        train=False,
        transform=test_transforms,
        target_transform=None
    )
#weights=returnqz()
#sample=WeightedRandomSampler(weights,len(test_dataset),replacement=True)
def returntest_dataloader(batch_size):
    return DataLoader(test_dataset,batch_size=batch_size,num_workers=0,shuffle=False)
