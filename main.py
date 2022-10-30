import os
import time
import torch
import numpy as np
import torch.nn as nn
from transformers import BertForSequenceClassification

import train_or_test
from config import Config
from models.BertClassification import Model
from utils import SentimentDataLoader

if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()
    print('--------------------------------------')
    print('start to load data ...')
    loader = SentimentDataLoader(config)
    train_dataloader = loader.get_train()
    dev_dataloader = loader.get_dev()
    test_dataloader = loader.get_test()
    print(len(train_dataloader))
    print(len(dev_dataloader))
    print(len(test_dataloader))
    loader_list = [train_dataloader, dev_dataloader, test_dataloader]
    model = Model(config)
    # model = BertForSequenceClassification.from_pretrained(config.checkpoint, num_labels=2, output_attentions=False,
    #                                                       output_hidden_states=True)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    print('--------------------------------------')
    print('start to train data ...')
    train_or_test.train(model, criterion, loader_list, config)
    print('--------------------------------------')
    print('start to test data ...')
    train_or_test.test(config, model, criterion, test_dataloader)
    # # ===================================
    for root, dirs, files in os.walk('./data/test_data'):
        for file in files:
            path = os.path.join(root, file)
            if path.find('movie') > 0:
                print(path)
                tmp_loader = loader.get_tmp(path)
                train_or_test.test(config, model, criterion, tmp_loader)
#     ======================================
    path= "/home/wangshuoxin/wsx/hugging_learn/data/test_data/movie_1.txt"
    tmp_loader = loader.get_tmp(path)
    train_or_test.test(config, model, criterion, tmp_loader)

