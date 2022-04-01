import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SimpleCustomModel(nn.Module):
    def __init__(self, config, output_hidden_states=False):
        super(SimpleCustomModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=output_hidden_states)
        self.model = AutoModel.from_pretrained(self.config.model_name, output_hidden_states=output_hidden_states)
        self.in_feautres = self.model_config.hidden_size
        self.fc = nn.Linear(self.in_feautres, 1)

        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, ids, mask):
        outputs = self.model(ids, attention_mask=mask)
        output = self.fc(outputs[1])
        return output