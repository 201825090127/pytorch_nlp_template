import os
import time

import numpy as np
from sklearn import metrics
from transformers import AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
import train_or_test
import utils
from evaluate import Eval


def train(model, criterion, loader, config):
    start_time = time.time()
    train_loader, dev_loader, _ = loader
    # 拿到model的所有参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(model.parameters(), lr=config.lr,
                      correct_bias=False)  # 要重现BertAdam特定的行为，请设置correct_bias = False
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=len(train_loader) * config.num_epochs)  # PyTorch调度程序用法如下：
    total_batch = 0  # 记录多少个batch
    dev_best_loss = float('inf')  # 记录校验集合最好的loss
    last_improve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练
    eval_tool = Eval(config)
    for epoch in range(config.num_epochs):
        # 启动BatchNormalization和dropout
        model.train()
        print('Epoch {}/{}'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_loader):
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            outputs = model(trains)  # 调用forward
            model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:  # 每多少次输出在训练集和校验集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss =eval_tool.evaluate(model, criterion, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_idf = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6},Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss{3:>5.2}, Val Acc{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_idf, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 在验证集上loss超过1000batch没有下降，结束训练
                print("在校验数据集合上已经很长时间没有提升了，模型自动停止训练")
                flag = True
                break
        if flag:
            break


def test(config, model, criterion,test_iter):
    '''
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    '''
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    eval_tool = Eval(config)

    # test_acc,test_loss= eval_tool.evaluate(model,criterion, test_iter)
    test_acc, test_loss, test_confusion =  eval_tool.evaluate(model,criterion, test_iter,test=True)
    msg = 'Test Loss:{0:>5.2},Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision Recall and F1-score ")
    # print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)
