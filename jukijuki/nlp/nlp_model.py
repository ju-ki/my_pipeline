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


class AttentionWeightModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_feautres = self.model_config.hidden_size
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, ids, mask):
        outputs = self.model(ids, attention_mask=mask)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        output = self.fc(self.fc_dropout(feature))
        return output


class MeanPoolingModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_feautres = self.model_config.hidden_size
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, ids, mask):
        outputs = self.model(ids, mask)
        last_hidden_state = outputs[0]
        attention_mask = mask

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        feature = sum_embeddings / sum_mask
        return feature

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        output = self.fc(self.fc_dropout(feature))
        return output


class ClsEmbeddingModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_feautres = self.model_config.hidden_size
        self.layer_norm = nn.LayerNorm(self.model_config.hidden_size)
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)
        self._init_weights(self.layer_norm)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, ids, mask):
        outputs = self.model(ids, mask)
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        feature = last_hidden_states[:, 0]
        return feature

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        output = self.fc(self.fc_dropout(feature))
        return output


class Second2LastLayerModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_feautres = self.model_config.hidden_size
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, ids, mask):
        outputs = self.model(ids, mask)
        all_hidden_states = torch.stack(outputs[1])
        layer_index = 11
        feature = all_hidden_states[layer_index+1, :, 0]
        return feature

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        output = self.fc(self.fc_dropout(feature))
        return output


class ConcatenatePoolingModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_feautres = self.model_config.hidden_size
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size * 4, self.config.target_size)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, ids, mask):
        outputs = self.model(ids, mask)
        all_hidden_states = torch.stack(outputs[1])
        concatenate_pooling = torch.cat((all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1)
        feature = concatenate_pooling[:, 0]
        return feature

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        output = self.fc(self.fc_dropout(feature))
        return output


class CNNPoolingModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        if self.config.TRAIN:
            self.model_config = AutoConfig.from_pretrained(self.config.model_name, output_hidden_states=True)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config.model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)
        self.in_features = self.model_config.hidden_size
        self.cnn1 = nn.Conv1d(self.in_features, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)
        self.fc_dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, ids, mask):
        outputs = self.model(ids, mask)
        last_hidden_state = outputs[0].permute(0, 2, 1)
        return last_hidden_state

    def forward(self, ids, mask):
        feature = self.feature(ids, mask)
        cnn_embeddings = F.relu(self.cnn1(feature))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        output, _ = torch.max(cnn_embeddings, 2)
        return output