import os
import torch
from torch.autograd import Variable
from functions.F1score import *
from functions.lossfunction import *
def train(model,traindata,testdata,args,optimizer,use_cuda):
    if not os.path.exists(args.save_dir):  # default=os.path.join('.', 'checkpoint')
        os.makedirs(args.save_dir)
    logfile=open(os.path.join(args.save_dir, args.log_file), 'a+')
    epoch_num=args.epoch_num
    Loss=FocalLoss()
    for epoch in range(epoch_num):
        for i, data in enumerate(traindata):
            model.train(True)
            image,label=data
            optimizer.zero_grad()
            if use_cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda()).long()
            else:
                image = Variable(image)
                label = Variable(label).long()   
            out,dirloss=model(image)
            loss=Loss(out,label)+dirloss
            if use_cuda:
                loss = loss.cuda()
            loss.backward()
            optimizer.step()
            print("第"+str(epoch)+"epoch --- 第"+str(i)+"轮"+"loss="+str(loss.data.item()))
        if (epoch + 1) % 5== 0 or epoch>90:
            macro_F1,acc = test(model=model, dataloader=testdata,use_cuda=use_cuda)
            print(str(macro_F1.data.item())+'  acc= '+str(acc))
            with open(os.path.join(args.save_dir, args.log_file), 'a+') as log_file:
                logfile.write('epoch ' + str(epoch) + '  macro_F1=' + str(macro_F1.data.item()) +'  acc= '+str(acc) +'\n')
        if epoch==199:
            save_path = os.path.join(args.save_dir,'epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_path)
        if use_cuda:
            torch.cuda.empty_cache()