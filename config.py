import argparse
import torch
import os
import random
import json
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'BertClassification'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.save_path =  os.path.join( self.model_dir, self.model_name+'.pkl')
        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir', type=str,
                            default='./data',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--seed', type=int,
                            default=5782,
                            help='random seed')
        parser.add_argument('--cuda', type=int,
                            default=0,
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--num_epochs', type=int,
                            default=100,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--batch_size', type=int,
                            default=128,
                            help='batch size')
        parser.add_argument('--lr', type=float,
                            default=1e-5,
                            help='learning rate')
        parser.add_argument('--padding_size', type=int,
                            default=64,
                            help='max length of sentence')
        parser.add_argument('--train_path', type=str,
                            default='./data/train.txt',
                            help='train data path')

        parser.add_argument('--dev_path', type=str,
                            default='./data/dev.txt',
                            help='dev data path')
        parser.add_argument('--test_path', type=str,
                            default='./data/test.txt',
                            help='test data path')

        parser.add_argument('--hidden_size', type=int,
                            default=768,
                            help='bert隐藏层个数')

        parser.add_argument('--checkpoint', type=str,
                            default='bert-base-uncased',
                            help='bert 模型')
        parser.add_argument('--require_improvement', type=int,
                            default=1000,
                            help='若超过1000batch效果还没提升，提前结束训练')
        parser.add_argument('--num_classes', type=int,
                            default=10,
                            help='类别数')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
