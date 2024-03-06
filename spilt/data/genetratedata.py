import os
import random
"""
    This code is for reference only.
    This code generates information related to the dataset and consolidates this information into three txt files. 
During training, rely on these three files to locate the images in the dataset.
    This code displays the information for forming the R1-14G600(big) dataset
    Divide the dataset according to specific circumstances during use
"""


def generate(dir, order,label):
    files = os.listdir(dir)
    files.sort()
    print('****************')
    print('input :', dir)
    print('start...')

    listText = open('order_imagename.txt', 'a+')
    listText1 = open('order_label.txt', 'a+')
    listText2 = open('train_test_split2.txt', 'a+')#
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = str(int(order)) + ' '+folder + '/' + file+'\n'
        name1 = str(int(order)) + ' '+str(int(label))+  '\n'
        if order<501 :
            name2 = str(int(order)) + ' ' + '1' + '\n'
        elif order<1501:
            name2 = str(int(order)) + ' ' + '0' + '\n'
        elif order<34301:
            name2 = str(int(order)) + ' ' + '3' + '\n'
        else:
            a=random.random()
            if a < 0.7:
                name2 = str(int(order)) + ' ' + '1' + '\n'
            else:
                name2 = str(int(order)) + ' ' + '0' + '\n'
        listText.write(name)
        listText1.write(name1)
        listText2.write(name2)
        order+=1
    listText.close()
    listText1.close()
    listText2.close()
    print('down!')
    print('****************')
    return order

outer_path = 'image'

if __name__ == '__main__':
    i = 1
    j=1
    folderlist = os.listdir(outer_path)
    for folder in folderlist:
        i=generate(os.path.join(outer_path, folder), i,j)
        j+= 1


