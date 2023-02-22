from torch.autograd import Variable
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings('ignore')
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rd
import numpy as np


class OurModel(torch.nn.Module):
    def __init__(self, data_config, pretrain_data, args, device):
        super(OurModel, self).__init__()
        self.pretrain_data = pretrain_data
        self.n_users_1 = data_config['n_users_1']
        self.n_users_2 = data_config['n_users_2']
        self.n_overlap_users = data_config['n_users_1']
        self.n_items_1 = data_config['n_items_1']
        self.n_items_2 = data_config['n_items_2']
        self.device = device
        self.ssl_temp=0.1

        self.keep_prob = args.keep_prob
        self.A_split = args.A_split

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.n_factors = args.n_factors
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.dropout = False

        self.alpha = args.alpha
        self.beta = args.beta

        self.folds = 10

        self.norm_adj_1 = data_config['norm_adj_1']
        self.norm_adj_2 = data_config['norm_adj_2']

        if not self.pretrain_data:
            self.embedding_dict = nn.ParameterDict({
                'user_embedding_1': nn.Parameter(nn.init.normal_(torch.empty(self.n_users_1, self.emb_dim), std=0.1)),
                'user_embedding_2': nn.Parameter(nn.init.normal_(torch.empty(self.n_users_2, self.emb_dim), std=0.1)),
                'item_embedding_1': nn.Parameter(nn.init.normal_(torch.empty(self.n_items_1, self.emb_dim), std=0.1)),
                'item_embedding_2': nn.Parameter(nn.init.normal_(torch.empty(self.n_items_2, self.emb_dim), std=0.1))
            })
        else:
            self.embedding_dict = nn.ParameterDict({
                'user_embedding_1': nn.Parameter(self.pretrain_data['user_embed_1']),
                'user_embedding_2': nn.Parameter(self.pretrain_data['user_embed_2']),
                'item_embedding_1': nn.Parameter(self.pretrain_data['item_embed_1']),
                'item_embedding_2': nn.Parameter(self.pretrain_data['item_embed_2'])
            })
            print('using pretrained initialization')

        self.f = nn.Sigmoid()
        self.linear = nn.Linear(self.emb_dim, self.emb_dim)
        self._init_graph(device)

    def _split_A_hat(self, A, n_users, n_items, device):
        A_fold = []
        fold_len = (n_users + n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = n_users + n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end], device).coalesce().to(device))
        return A_fold
    
    def _convert_sp_mat_to_sp_tensor(self, X, device):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long().to(device)
        col = torch.Tensor(coo.col).long().to(device)
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data).to(device)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).to(device)

    def _init_graph(self, device):
        if self.A_split:
            self.Graph_1 = self._split_A_hat(self.norm_adj_1, self.n_users_1, self.n_items_1, self.device)
            self.Graph_2 = self._split_A_hat(self.norm_adj_2, self.n_users_2, self.n_items_2, self.device)
        else:
            self.Graph_1 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_1, self.device)
            self.Graph_1 = self.Graph_1.coalesce().to(device)
            self.Graph_2 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_2, self.device)
            self.Graph_2 = self.Graph_2.coalesce().to(device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob, ori_graph):
        if self.A_split:
            graph = []
            for g in ori_graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(ori_graph, keep_prob)
        return graph

    def computer(self, graph, users_emb, items_emb, n_users, n_items):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob, graph)
            else:
                g_droped = graph   
        else:
            g_droped = graph

        layer_embs = []
        factor_num = [self.n_factors for i in range(self.n_layers)]
        for layer in range(self.n_layers):
            n_factors_l = factor_num[layer]
            all_embs_tp = torch.split(all_emb, int(self.emb_dim/n_factors_l), 1)
            all_embs = []
            for i in range(n_factors_l):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_embs_tp[i]))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_embs.append(side_emb)
                else:
                    all_embs.append(torch.sparse.mm(g_droped, all_embs_tp[i]))
            layer_embs.append(all_embs)
            factor_embedding = torch.cat([all_embs[0], all_embs[1]], dim=1)
            embs.append(factor_embedding)
            all_emb = factor_embedding
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [n_users, n_items])
        return users, items, layer_embs[-1]
            
    def create_bpr_loss(self, users, pos_items, neg_items, users_pre, pos_items_pre, neg_items_pre):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        regularizer = (torch.norm(users_pre) ** 2 + torch.norm(pos_items_pre) ** 2 + 
                       torch.norm(neg_items_pre) ** 2) / 2
        regularizer = regularizer / self.batch_size
        
        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss
    
    def calc_ssl_loss_strategy(self, layer_embed_1, layer_embed_2, n_users_1, n_users_2):
        invariant_embed_1, specific_embed_1 = layer_embed_1[0], layer_embed_1[1]
        invariant_u_embed_1, specific_u_embed_1 = invariant_embed_1[:n_users_1], specific_embed_1[:n_users_1]
        invariant_embed_2, specific_embed_2 = layer_embed_2[0], layer_embed_2[1]
        invariant_u_embed_2, specific_u_embed_2 = invariant_embed_2[:n_users_2], specific_embed_2[:n_users_2]
        normalize_invariant_user_1 = torch.nn.functional.normalize(invariant_u_embed_1, p=2, dim=1)
        normalize_invariant_user_2 = torch.nn.functional.normalize(invariant_u_embed_2, p=2, dim=1)

        normalize_specific_user_1 = torch.nn.functional.normalize(specific_u_embed_1, p=2, dim=1)
        normalize_specific_user_2 = torch.nn.functional.normalize(specific_u_embed_2, p=2, dim=1)

        pos_score_user = torch.sum(torch.mul(normalize_invariant_user_1, normalize_invariant_user_2), dim=1)

        neg_score_1 = torch.sum(torch.mul(normalize_invariant_user_1, normalize_specific_user_1), dim=1)
        neg_score_2 = torch.sum(torch.mul(normalize_invariant_user_2, normalize_specific_user_2), dim=1)
        neg_score_3 = torch.sum(torch.mul(normalize_specific_user_1, normalize_specific_user_2), dim=1)

        neg_score_4 = torch.matmul(normalize_invariant_user_1, normalize_invariant_user_2.T)

        pos_score = torch.exp(pos_score_user / self.ssl_temp)
        neg_score_1 = torch.exp(neg_score_1 / self.ssl_temp)
        neg_score_2 = torch.exp(neg_score_2 / self.ssl_temp)
        neg_score_3 = torch.exp(neg_score_3 / self.ssl_temp)
        neg_score_4 = torch.sum(torch.exp(neg_score_4 / self.ssl_temp), dim=1)

        ssl_loss_user = -torch.sum(torch.log(pos_score / (neg_score_1 + neg_score_2 + neg_score_3 + pos_score +
                                                          neg_score_4)))
        ssl_loss = ssl_loss_user
        return ssl_loss
    
    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.sigmoid(torch.sum(u_g_embeddings*i_g_embeddings, axis=1, keepdim=True))
    
    def get_embeddings(self, users_1, pos_items_1, neg_items_1, users_2, pos_items_2, neg_items_2):
        self.u_g_embeddings_1 = torch.index_select(self.ua_embeddings_1, 0, torch.LongTensor(users_1).to(self.device))
        self.pos_i_g_embeddings_1 = torch.index_select(self.ia_embeddings_1, 0, torch.LongTensor(pos_items_1).to(self.device))
        self.neg_i_g_embeddings_1 = torch.index_select(self.ia_embeddings_1, 0, torch.LongTensor(neg_items_1).to(self.device))
        
        self.u_g_embeddings_2 = torch.index_select(self.ua_embeddings_2, 0, torch.LongTensor(users_2).to(self.device))
        self.pos_i_g_embeddings_2 = torch.index_select(self.ia_embeddings_2, 0, torch.LongTensor(pos_items_2).to(self.device))
        self.neg_i_g_embeddings_2 = torch.index_select(self.ia_embeddings_2, 0, torch.LongTensor(neg_items_2).to(self.device))
        
    def forward(self, users_1, pos_items_1, neg_items_1, users_2, pos_items_2, neg_items_2):
        self.ua_embeddings_1, self.ia_embeddings_1, self.layer_embeddings_1 = \
            self.computer(self.Graph_1, self.embedding_dict['user_embedding_1'], self.embedding_dict['item_embedding_1'], self.n_users_1, self.n_items_1)
        self.ua_embeddings_2, self.ia_embeddings_2, self.layer_embeddings_2 = \
            self.computer(self.Graph_2, self.embedding_dict['user_embedding_2'], self.embedding_dict['item_embedding_2'], self.n_users_2, self.n_items_2)
        
        self.u_g_embeddings_1 = torch.index_select(self.ua_embeddings_1, 0, torch.LongTensor(users_1).to(self.device))
        self.pos_i_g_embeddings_1 = torch.index_select(self.ia_embeddings_1, 0, torch.LongTensor(pos_items_1).to(self.device))
        self.neg_i_g_embeddings_1 = torch.index_select(self.ia_embeddings_1, 0, torch.LongTensor(neg_items_1).to(self.device))
        self.u_g_embeddings_pre_1 = torch.index_select(self.embedding_dict['user_embedding_1'], 0, torch.LongTensor(users_1).to(self.device))
        self.pos_i_g_embeddings_pre_1 = torch.index_select(self.embedding_dict['item_embedding_1'], 0, torch.LongTensor(pos_items_1).to(self.device))
        self.neg_i_g_embeddings_pre_1 = torch.index_select(self.embedding_dict['item_embedding_1'], 0, torch.LongTensor(neg_items_1).to(self.device))

        self.u_g_embeddings_2 = torch.index_select(self.ua_embeddings_2, 0, torch.LongTensor(users_2).to(self.device))
        self.pos_i_g_embeddings_2 = torch.index_select(self.ia_embeddings_2, 0, torch.LongTensor(pos_items_2).to(self.device))
        self.neg_i_g_embeddings_2 = torch.index_select(self.ia_embeddings_2, 0, torch.LongTensor(neg_items_2).to(self.device))
        self.u_g_embeddings_pre_2 = torch.index_select(self.embedding_dict['user_embedding_2'], 0, torch.LongTensor(users_2).to(self.device))
        self.pos_i_g_embeddings_pre_2 = torch.index_select(self.embedding_dict['item_embedding_2'], 0, torch.LongTensor(pos_items_2).to(self.device))
        self.neg_i_g_embeddings_pre_2 = torch.index_select(self.embedding_dict['item_embedding_2'], 0, torch.LongTensor(neg_items_2).to(self.device))

        mf_loss_1, emb_loss_1 = self.create_bpr_loss(self.u_g_embeddings_1, 
                                                     self.pos_i_g_embeddings_1, 
                                                     self.neg_i_g_embeddings_1,
                                                     self.u_g_embeddings_pre_1,
                                                     self.pos_i_g_embeddings_pre_1, 
                                                     self.neg_i_g_embeddings_pre_1)
        mf_loss_2, emb_loss_2 = self.create_bpr_loss(self.u_g_embeddings_2, 
                                                     self.pos_i_g_embeddings_2, 
                                                     self.neg_i_g_embeddings_2,
                                                     self.u_g_embeddings_pre_2,
                                                     self.pos_i_g_embeddings_pre_2, 
                                                     self.neg_i_g_embeddings_pre_2)
        ssl_loss = self.calc_ssl_loss_strategy(self.layer_embeddings_1, self.layer_embeddings_2, self.n_users_1, self.n_users_2)
        loss = mf_loss_1 + mf_loss_2 + self.alpha * ssl_loss
        return loss
