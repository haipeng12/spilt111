import torch
from torch.autograd import Variable


def test(model,dataloader,use_cuda=True):
    class_num = 7
    model.train(False)
    with torch.no_grad():
        cur_step = 0
        macro_F1=torch.zeros(7)
        num=len(dataloader.dataset)
        scorenum = torch.zeros(7, 3).int()
        correct=0
        for batch_no, data in enumerate(dataloader):
            cur_step += 1
            image, label = data
            if use_cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda()).long()
            else:
                image = Variable(image)
                label = Variable(label).long()
            out,_ = model(image)
            for i in range(label.shape[0]):
                _, index = torch.topk(out[i], 1)
                index=index.data.item()
                if index==label[i]:
                    scorenum[label[i]][0]+=1  
                else:
                    scorenum[index][1]+=1      
                    scorenum[label[i]][2]+=1 
        for i in range(class_num):
            p=scorenum[i][0]/(scorenum[i][0]+scorenum[i][1])
            r=scorenum[i][0]/(scorenum[i][0]+scorenum[i][2])

            if p==0 or r==0:
                macro_F1[i]=0
            else:
                macro_F1[i]=2*p*r/(p+r)
            correct+=scorenum[i][0]
        print(macro_F1)
    return macro_F1.mean(dim=0),correct/num #Macro-F1, acc
