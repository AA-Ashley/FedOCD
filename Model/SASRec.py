from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from Model.BasicModel import BasicModel


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, parser, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.parser = parser
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(parser.device)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate).to(parser.device)
        self.relu = torch.nn.ReLU().to(parser.device)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(parser.device)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate).to(parser.device)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(BasicModel):
    def __init__(self, parser):
        super(SASRec, self).__init__(parser)
        self.emb_pos = torch.nn.Embedding(parser.seq, parser.emb_size).to(parser.device)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=parser.dropout).to(parser.device)

        self.attentionLayerNorms = torch.nn.ModuleList().to(parser.device)
        self.attentionLayers = torch.nn.ModuleList().to(parser.device)
        self.forwardLayerNorms = torch.nn.ModuleList().to(parser.device)
        self.forwardLayers = torch.nn.ModuleList().to(parser.device)

        self.lastLayerNorm = torch.nn.LayerNorm(parser.emb_size, eps=1e-8).to(parser.device)
        self.logistic = torch.nn.Sigmoid().to(parser.device)

        for _ in range(parser.num_blocks):
            newAttentionLayerNorm = torch.nn.LayerNorm(parser.emb_size, eps=1e-8).to(parser.device)
            self.attentionLayerNorms.append(newAttentionLayerNorm)

            newAttentionLayer = torch.nn.MultiheadAttention(parser.emb_size, parser.num_heads, parser.dropout).to(parser.device)
            self.attentionLayers.append(newAttentionLayer)

            newForwardLayerNorm = torch.nn.LayerNorm(parser.emb_size, eps=1e-8).to(parser.device)
            self.forwardLayerNorms.append(newForwardLayerNorm)

            newForwardLayer = PointWiseFeedForward(parser, parser.emb_size, parser.dropout).to(parser.device)
            self.forwardLayers.append(newForwardLayer)

        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # just ignore those failed init layers

    def computer(self, log_seqs):
        seqs = self.emb_item(log_seqs)
        seqs *= self.emb_item.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.emb_pos(torch.LongTensor(positions).to(self.parser.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        attention_mask = attention_mask.to(self.parser.device)

        for i in range(len(self.attentionLayers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attentionLayerNorms[i](seqs)
            mhaOutputs, _ = self.attentionLayers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mhaOutputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forwardLayerNorms[i](seqs)
            seqs = self.forwardLayers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        # (U, T, C) -> (U, -1, C)
        log_feats = self.lastLayerNorm(seqs)

        return log_feats

    def forward(self, users, pos_items, neg_items, seqs):
        log_feats = self.computer(seqs)[:, -1, :]
        emb_pos = self.emb_item(pos_items)
        emb_neg = self.embedItem(neg_items)
        pos_score = (log_feats * emb_pos).sum(dim=-1)
        neg_score = (log_feats * emb_neg).sum(dim=-1)
        return pos_score, neg_score

    def get_loss(self, users, pos_items, neg_items, seqs):
        pos_scores, neg_scores = self.forward(users, seqs, pos_items, neg_items)
        loss = - (pos_scores - neg_scores).sigmoid().log().mean()
        return loss, 0

    def predict(self, users, items, seqs):
        log_feats = self.computer(seqs)
        final_feat = log_feats[:, -1, :]
        self.embedUser.weight.data[users.long()] = final_feat
        # user_ids hasn't been used yet
        # only use last QKV classifier, a waste
        # (U, I, C)
        emb_items = self.emb_item(items)
        emb_users = self.emb_user(users)
        scores = torch.mul(emb_users, emb_items)
        scores = torch.sum(scores, dim=1)
        scores = self.logistic(scores)
        return scores
