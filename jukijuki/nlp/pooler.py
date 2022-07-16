import torch
import numpy as np
import torch.nn as nn


class MeanPoolingV1(nn.Module):
    def __init__(self):
        super(MeanPoolingV1, self).__init__()

    def forward(self, outputs, mask=None):
        last_hidden_states = outputs[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPoolingV1(nn.Module):
    def __init__(self):
        super(MaxPoolingV1, self).__init__()

    def forward(self, outputs, mask=None):
        last_hidden_states = outputs[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        last_hidden_states[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_states, 1)[0]
        return max_embeddings


class MeanMaxPoolingV1(nn.Module):
    def __init__(self):
        super(MeanMaxPoolingV1, self).__init__()

    def forward(self, outputs, mask=None):
        last_hidden_states = outputs[0]
        mean_pooling_embeddings = torch.mean(last_hidden_states, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_states, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings


class CNNPoolingV1(nn.Module):
    """_summary_
    example_usage:
            self.pooler = CNNPoolingV1()
            self.cnn1 = nn.Conv1d(758, 256, kernel_size=2, padding=1)
            self.cnn2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)

        def forward(ids, mask):
            last_hidden_states = self.model(ids, mask)[0]
            last_hidden_states = self.pooler(last_hidden_states)
            cnn_embeddings = F.relu(self.cnn1(last_hidden_states))
            cnn_embeddings = self.cnn2(cnn_embedding)
            logits, _ = torch.max(cnn_embeddings, 2)
    """
    def __init__(self):
        super(CNNPoolingV1, self).__init__()

    def forward(self, outputs, mask=None):
        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.permute(0, 2, 1)
        return last_hidden_states


class CLSEmbeddingPoolingV1(nn.Module):
    def __init__(self):
        super(CLSEmbeddingPoolingV1, self).__init__()

    def forward(self, outputs, mask=None):
        last_hidden_states = outputs[0]
        return last_hidden_states[:, 0]


class SecondToLastPoolingV1(nn.Module):
    def __init__(self, layer_index):
        super(SecondToLastPoolingV1, self).__init__()
        self.layer_index = layer_index

    def forward(self, outputs, mask=None):
        hidden_states = torch.stack(outputs["hidden_states"])
        cls_embeddings = hidden_states[self.layer_index+1, :, 0]
        return cls_embeddings


class ConcatenatePoolingV1(nn.Module):
    def __init__(self, num):
        super(ConcatenatePoolingV1, self).__init__()
        self.num = num

    def forward(self, outputs, mask=None):
        hidden_states = outputs["hidden_states"]
        concatenate_pooling = torch.cat([hidden_states[-1*i][:,0] for i in range(1, self.num+1)], dim=1) 
        return concatenate_pooling


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


class WeightedLayerPoolingV1(nn.Module):
    def __init__(self, num_hidden_layer, layer_start, layer_weights=None):
        super(WeightedLayerPoolingV1, self).__init__()
        self.num_hidden_layer = num_hidden_layer
        self.layer_start = layer_start
        self.layer_weights = layer_weights
        self.pooler = WeightedLayerPooling(num_hidden_layers=self.num_hidden_layer, layer_start=self.layer_start, layer_weights=self.layer_weights)

    def forward(self, outputs, mask=None):
        hidden_states = torch.stack(outputs["hidden_states"])
        weighted_pooling_embeddings = self.pooler(hidden_states)
        weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        return weighted_pooling_embeddings


class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
        self._init_weights(self.attention)

    def forward(self, x, mask):
        w = self.attention(x).float()
        w[mask==0]=float('-inf')
        w = torch.softmax(w,1)
        x = torch.sum(w * x, dim=1)
        return x

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


class AttentionPoolingV1(nn.Module):
    def __init__(self, dim1, dim2):
        super(AttentionPoolingV1, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.Tanh(),
            nn.Linear(dim2, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, outputs, mask=None):
        hidden_states = outputs[0]
        weights = self.attention(hidden_states)
        feature = torch.sum(weights * hidden_states, dim=1)
        return feature


class AttentionPoolingV2(nn.Module):
    def __init__(self, in_dim):
        super(AttentionPoolingV2, self).__init__()
        self.in_dim = in_dim
        self.pooler = AttentionPool(self.in_dim)

    def forward(self, outputs, mask=None):
        hidden_states = outputs[0]
        feature = self.pooler(hidden_states, mask)
        return feature


class AttentionPoolingV3(nn.Module):
    def __init__(self, in_dim):
        super(AttentionPoolingV3, self).__init__()
        self.in_dim = in_dim
        self.pooler = AttentionPool(in_dim)

    def forward(self, outputs, mask=None):
        hidden_states = outputs[0]
        hidden_states1 = hidden_states[-1]
        hidden_states2 = hidden_states[-2]
        last_hidden_states = torch.cat((hidden_states1, hidden_states2), 2)
        feature = self.pooler(last_hidden_states, mask)
        return feature


class WKPooling(nn.Module):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, all_hidden_states, mask):
        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1,0)
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]  # Start from 4th layers output

        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = mask.cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1  # Not considering the last item
        embedding = []

        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            ##features.update({'sentence_embedding': features['cls_token_embeddings']})

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):
        ## Unify Token Representation
        window_size = self.context_window_size

        alpha_alignment = torch.zeros(token_feature.size()[0], device=token_feature.device)
        alpha_novelty = torch.zeros(token_feature.size()[0], device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size:k, :]
            right_window = token_feature[k + 1:k + window_size + 1, :]
            window_matrix = torch.cat([left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.qr(window_matrix.T)

            r = R[:, -1]
            alpha_alignment[k] = torch.mean(self.norm_vector(R[:-1, :-1], dim=0), dim=1).matmul(R[:-1, -1]) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):
        ## Implements the normalize() function from sklearn
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):
        ## Unify Sentence By Token Importance
        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


class WKPoolingV1(nn.Module):
    def __init__(self):
        super(WKPoolingV1, self).__init__()
        self.pooler = WKPooling(layer_start=9)

    def forward(self, outputs, mask=None):
        hidden_states = torch.stack(outputs["hidden_states"])
        wkpooling_embeddings = self.pooler(hidden_states, mask)
        return wkpooling_embeddings


class LSTMPoolingV1(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm, dropout_rate):
        super(LSTMPoolingV1, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, outputs, mask=None):
        all_hidden_states = torch.stack(outputs["hidden_states"])
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class TransformerPoolingV1(nn.Module):
    def __init__(self, in_features, max_length, num_layers=1, nhead=8, num_targets=1):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=in_features,
                                                                                          nhead=nhead),
                                                 num_layers=num_layers)
        self.row_fc = nn.Linear(in_features, 1)
        self.out_features = max_length

    def forward(self, outputs, mask=None):
        last_hidden_state = outputs["last_hidden_state"]
        out = self.transformer(last_hidden_state)
        out = self.row_fc(out).squeeze(-1)
        return out