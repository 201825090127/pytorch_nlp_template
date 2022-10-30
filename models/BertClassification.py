import torch
import torch.nn as nn
from transformers import AutoModel

from config import Config


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.checkpoint)
        for param in self.parameters():
            param.requires_grad = True  # 参数微调
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        bert_output = self.encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        res = self.fc(cls_vectors)
        return res
# if __name__ =='__main__':
#     config=Config()
#     model=Model(config)
#     model=model.to(config.device)
#     print(model)
