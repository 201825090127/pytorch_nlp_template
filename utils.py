from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm  # 进度条库
import time
import torch
from datetime import timedelta
from transformers import AutoTokenizer

from config import Config


class SentimentDataset(Dataset):
    def __init__(self, file_path):
        self.path = file_path
        self.dataset, self.label = self.__load_data()

    def __load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data_list = f.readlines()
            datas = [data.strip().split('\t')[0] for data in data_list]
            labels = [int(data.strip().split('\t')[1]) for data in data_list]
            f.close()
        return datas, labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.dataset[index], self.label[index]


class SentimentDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

    def __collate_fn(self, batch):
        data, labels = zip(*batch)  # unzip the batch data
        data = list(data)
        labels = list(labels)
        X = self.tokenizer(data,  # 输入文本
                           add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                           max_length=self.config.padding_size,  # 填充 & 截断长度
                           padding='max_length',
                           return_attention_mask=True,  # 返回 attn. masks.
                           truncation=True,
                           return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                           )
        y = torch.tensor(labels)
        return X, y

    def __get_data(self, path, shuffle=True):
        dataset = SentimentDataset(path)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, collate_fn=self.__collate_fn)
        return loader

    def get_train(self):
        return self.__get_data(self.config.train_path)

    def get_dev(self):
        return self.__get_data(self.config.dev_path)

    def get_test(self):
        return self.__get_data(self.config.test_path)

    def get_tmp(self, path):
        return self.__get_data(path)


def get_time_dif(start_time):
    '''
    获取已经使用的时间
    :param start_time:
    :return:
    '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    config = Config()
    config.print_config()
    test_path = config.test_path
    test_data = SentimentDataset(test_path)
    print(test_data.__len__())
    text, label = test_data.__getitem__(0)
    print(text)
    print(label)
    loader = SentimentDataLoader(config)
    test_dataloader = loader.get_test()
    batch_X, batch_y = next(iter(test_dataloader))
    print("batch_X: ", {k: v.shape for k, v in batch_X.items()})
    print("batch_y: ", batch_y.shape)
    print(batch_X)
    print(batch_y)
