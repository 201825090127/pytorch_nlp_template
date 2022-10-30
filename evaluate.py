import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import dataloader
from tqdm import tqdm


class Eval(object):
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, criterion, dev_iter, test=False):
        '''
        :param config:
        :param model:
        :param dev_iter:
        :return:
        '''
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for X, y in dev_iter:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = model(X)
                loss = criterion(outputs, y)
                loss_total = loss_total + loss
                y = y.data.cpu().numpy()
                predict = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, y)
                predict_all = np.append(predict_all, predict)
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            class_list = ['fake','truth' ]
            # report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(dev_iter) , confusion
        return acc, loss_total / len(dev_iter)
