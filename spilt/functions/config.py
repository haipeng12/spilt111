import argparse
import os
import torch



def getArgs(ipython=False):
    parser = argparse.ArgumentParser()

    if ipython:
        parser.add_argument("-f", dest='j_cfile', type=str)

    parser.add_argument('--class_num', dest='class_num',
                        default=7, type=int)
    parser.add_argument('--savepoint_file', dest='savepoint_file',
                        default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        default=os.path.join('.', 'checkpoint5'), type=str) 
    parser.add_argument('--log_file', dest='log_file',
                        default='log1_2.txt', type=str)

    parser.add_argument('--epoch_num', dest='epoch_num',
                        default=200, type=int)
    parser.add_argument('--train_batch', dest='train_batch',
                        default=32, type=int)
    parser.add_argument('--test_batch', dest='test_batch',
                        default=64, type=int)
    parser.add_argument('--savepoint', dest='savepoint',
                        default=5000, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default=5000, type=int)
    parser.add_argument('--test_img_path', dest='test_img_path',
                        default=None, type=str)
    parser.add_argument('--test_branches', dest='test_branches',
                        default='0,1,2,3', type=str)
    parser.add_argument('--test_save_path', dest='test_save_path',
                        default=os.path.join('.', 'vis'), type=str)

    # LACK OF TRANSFORMATION-RELATED ARGUMENTS

    args = parser.parse_args()

    return args