import torch
from torch import nn

from data.dataset import *
import torch.optim as optim
from backbone.SpiltAttresnet import *
#from backbone.ShuSpiltresnet import *
from functions.trainfunction import*
from functions.config import *

if __name__ == '__main__':
    args= getArgs()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #Import Model
    model = remodel()
    #arcloss=Arcloss()
    if use_cuda:
        model = model.cuda()
        #arcloss=arcloss.cuda()
    if torch.cuda.device_count() > 1:
        model = model.to(torch.device('cuda:0'))
    #data
    traindata=returntrain_dataloader(args.train_batch)
    test_dataset=returntest_dataloader(args.test_batch)
    optimizer=optim.Adam(model.parameters(), lr=0.0001,weight_decay = 5e-4)
    milestones = [80,120]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.3)

    train(model,traindata,test_dataset,scheduler,args,optimizer,use_cuda=use_cuda)