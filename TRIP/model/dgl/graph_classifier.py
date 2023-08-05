from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from .discriminator import Discriminator
from .batch_gru import BatchGRU
import networkx as nx
import dgl
# from .ori_path import all_simple_edge_paths
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id, ent2rels):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.ent2rels = ent2rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        # num_rels + 1 instead of nums_rels, in order to add a "padding" relation.
        self.rel_emb = nn.Embedding(self.params.num_rels + 1, self.params.inp_dim, sparse=False, padding_idx=self.params.num_rels)
        # 以下为新增
        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.params.sem_dim).uniform_(-1, 1))
        if self.params.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.params.inp_dim, self.params.sem_dim)
        elif self.params.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.params.inp_dim, self.params.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.params.nei_rels_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)
        #########################
        if self.params.add_ht_emb:    
            # self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        #########################
        if self.params.comp_hrt:
            self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, 1)
        
        if self.params.nei_rel_path:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + 2 * self.params.emb_dim, 1)
            # self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + 3 * self.params.emb_dim, 1)
        if self.params.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)

        if self.params.nei_rel_path:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim)
        else:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim , self.params.num_gcn_layers * self.params.emb_dim)
        # 用来学习路径表示的模块
        # input_dim = self.params.emb_dim 实体的维度 跟关系维度值一致，hidden_dim = self.params.emb_dim
        # (batch,seq,feature)
        self.rnn = torch.nn.GRU(self.params.emb_dim, self.params.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.params.num_gcn_layers * self.params.emb_dim )

        self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim * 2, self.params.num_gcn_layers * self.params.emb_dim)

    def init_ent_emb_matrix(self, g):
        """ Initialize feature of entities by matrix form """
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)

        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        
        if self.params.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.params.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.params.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)  # [B, self.inp_dim]

    def comp_ht_emb(self, head_embs, tail_embs):
        if self.params.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.params.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.params.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_embs, tail_embs, rel_embs):
        rel_embs = rel_embs.repeat(1, self.params.num_gcn_layers)
        if self.params.comp_hrt == 'TransE':
            hrt_embs = head_embs + rel_embs - tail_embs
        elif self.params.comp_hrt == 'DistMult':
            hrt_embs = head_embs * rel_embs * tail_embs
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.comp_hrt} not recognized.')
        
        return hrt_embs
    # 正在改的地方。。。。。需要适配下size 然后喂到RNN来对路径表示进行学习
    def get_rel_path(self, subgraph, rel_labels, r_emb_out, max_paths = 200, max_path_len = 3, add_reverse=True):
        nodes = np.array(subgraph.nodes())
        edges = subgraph.edges() # class 'tuple'
        # print("nodes:",nodes)
        # print("edges:",edges)

        head_index = (subgraph.ndata['id'] == 1).nonzero().squeeze(1)
        # print("head_index:",head_index)
        # head_ids = edges[0][head_index]
        # print("head_ids:",head_ids)
        # tail_ids = (subgraph.ndata['id'] == 2).nonzero().squeeze(1)
        tail_index = (subgraph.ndata['id'] == 2).nonzero().squeeze(1)
        # tail_ids = edges[1][tail_index]

        edges = [np.array(x) for x in list(edges)]
        # edges = np.array(list(edges))
        # print(type(edges))
        # 'label' 代表关系id， type代表正负样例
        # print('imformation of edata type',subgraph.edata['type']) 
        # print('imformation of parent nid',subgraph.ndata['parent_id']) 
        edge_type = np.array(subgraph.edata['type'].cpu())
        # print("edge_type",edge_type)
        target_label = subgraph.edata['label'][0]

        type_edges = list(zip(edges[0], edges[1], edge_type))
        # print("type_edges shape:",len(type_edges))
        type_edges = [tuple(item) for item in type_edges]
        # for x in type_edges:
        #     print(x)
        #     break

        temp_nx = nx.MultiDiGraph(type_edges)
        # print("graph nodes info:",temp_nx.nodes.data)
        # print("temp_nx node information:",temp_nx.nodes.data)
        # print("temp_nx edge information:",temp_nx.edges.data)
        # print("subgraph.ndata['id']:",subgraph.ndata['id'])
        # head_ids = (subgraph.ndata['id'] == 1).nonzero().squeeze(1)
        # print('subgraph.ndata[id] shape:',subgraph.ndata['id'].shape)

        # print('imformation of one nodes',temp_nx.nodes[0])
        # if head_ids[0] not in temp_nx:
        #     print("????")
        # if not temp_nx.has_node(torch.tensor(0)):
        #     print("12112")
        batch_paths = []
        padding_rel = r_emb_out.shape[0] - 1
        for head_id, tail_id in zip(head_index,tail_index):
            # print(type(head_id))
            path_list = []
            path_num = 0
            # print(type(head_id))  class : tensor
            # print('关系路径:')  # 前两个就是目标节点，所以source = 0 , target = 1   [(1, 2, 'rel1'), (2, 3, 'rel2')] 
            for path in all_simple_edge_paths(temp_nx, source=int(head_id), target=int(tail_id), cutoff=max_path_len):
                path_num += 1
                # path_list.append(tuple([item[-1] for item in path]))
                if len(path) == 1 and path[0][-1] == target_label:  # 是待预测路径   似乎由BUG？？ 只要len==1 就可以pass了其实了
                    pass
                elif len(path) < max_path_len:
                    temp_path = [item[-1] for item in path]
                    temp_path.extend([padding_rel] * (max_path_len - len(path)))
                    # .extend([padding_rel] * (max_path_len - len(path)))
                    # print(temp_path)
                    path_list.append(tuple(temp_path))
                elif len(path) == max_path_len:
                    path_list.append(tuple([item[-1] for item in path]))
                if path_num > max_paths:  # 防止死循环
                    break
            # 去重 
            path_list = list(set(path_list))
            if len(path_list) == 0:
                # print(path_list)
                path_list.append(tuple([r_emb_out.shape[0] - 1]*max_path_len))
            batch_paths.append(path_list)
        # print(np.array(batch_paths).shape)
        # for x in batch_paths:
        #     print(x)
        # print("r_emb_out.shape[0]:",r_emb_out.shape[0])
        # shape : [10, 32]
        # print("shape:",r_emb_out.shape)
        # print(r_emb_out[9])
        # batch_paths : [batch, n_path, max_path_len]
        batch_size = np.array(batch_paths, dtype=object).shape[0]
        # print(batch_paths[0],batch_paths[1])
        for k,rel_path in enumerate(batch_paths):
            # paths_emb = torch.zeros((1,self.params.emb_dim))
            for i,path_i in enumerate(rel_path):
                path_i = torch.LongTensor(path_i).to(rel_labels.device)
                path_i_emb = F.embedding(path_i, r_emb_out, padding_idx=-1).unsqueeze(0) # [1,max_path_len,32]
                # print(path_i_emb)
                if i == 0:
                    paths_emb = path_i_emb
                else:
                    paths_emb = torch.cat((paths_emb,path_i_emb), dim=0)

            # print(paths_emb)
            _, last_state = self.rnn(paths_emb) # torch.Size([1, 4, 32])

            # print(last_state.shape)
            if self.params.path_agg == 'mean':
                output_i = torch.mean(last_state, 1) # [B, inp_dim]
            if self.params.path_agg == 'att':
                output_i = torch.mean(last_state, 1) # [B, inp_dim]
                # r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
                # atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
                # atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
                # output_i = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
            else:
                raise ValueError('unknown path_agg')
            
            if k == 0:
                output = output_i
            else:
                output = torch.cat((output_i,output), dim=0)
        # batch_size = batch_paths.shape[0]
        # batch_paths = batch_paths.view(batch_size * len(path_list), -1) # [B * n_paths, 3]
        # batch_paths = torch.LongTensor(batch_paths).to(rel_labels.device)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        # batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # # Input RNN 使用rnn 模块来对路径表示进行学习
        # _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        # last_state = last_state.squeeze(0) # squeeze the dim 0 
        # last_state = last_state.view(batch_size, len(path_list), self.params.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention 使用注意力机制来汇聚路径的表示，可以在这里改～
        # if self.params.path_agg == 'mean':
        #     output = torch.mean(last_state, 1) # [B, inp_dim]
        
        # if self.params.path_agg == 'att':
        #     r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
        #     atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
        #     atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
        #     output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        # else:
        #     raise ValueError('unknown path_agg')
        
        return output # [B, inp_dim]
    
    def get_context_graph(subgraph, max_paths = 200, max_path_len = 3, add_reverse=True):  # 4 是否增加逆关系 5 -> 10
        nodes = np.array(subgraph.nodes())
        edges = subgraph.edges()
        # print(nodes)
        # print(edges)
        edge_type = subgraph.edata['type']
        target_label = subgraph.edata['label'][0]

        type_edges = list(zip(edges[0], edges[1], edge_type))
        type_edges = np.array(type_edges)
        type_edges = [tuple(item) for item in type_edges]

        temp_nx = nx.MultiDiGraph(type_edges)
        path_list = []
        path_num = 0

        # print('关系路径:')  # 前两个就是目标节点，所以source = 0 , target = 1   [(1, 2, 'rel1'), (2, 3, 'rel2')] 
        for path in nx.all_simple_edge_paths(temp_nx, source=0, target=1, cutoff=max_path_len):
            path_num += 1
            if len(path) == 1 and path[0][-1] == target_label:  # 是待预测路径   似乎由BUG？？ 只要len==1 就可以pass了其实了
                pass
            elif len(path) <= max_path_len:
                path_list.append(tuple([item[-1] for item in path]))
            if path_num > max_paths:  # 防止死循环
                break
        # for path in path_list:
        #     print(path)

        # print('mmmm',len(path_list))
        path_list = list(set(path_list))
        path_len = len(path_list)
        relations = list(set(np.array(edge_type)))
        relation_len = len(relations)
        entity_list = list(nodes)
        # cgraph的节点个数 = 关系个数+路径个数+节点个数
        nodes_len = path_len + relation_len + len(entity_list)

        cg_nx = nx.MultiDiGraph()
        cg_nx.add_nodes_from(list(range(nodes_len)))
        # 按照path  relation entity的顺序标记节点
        rel_index_start = path_len
        ent_index_start = path_len + relation_len
        # Add edges
        nx_triplets = []
        for i, path in enumerate(path_list):
            for kk, rel in enumerate(path):  # 0 1 2
                rel_index = relations.index(rel) + rel_index_start
                nx_triplets.append((i, rel_index, {'type': kk}))  # path -> relation 这个type 表示是该关系路径的第几个关系
        for h,t,r in type_edges:
            rel_index = relations.index(r) + rel_index_start
            h = h + ent_index_start
            t = t + ent_index_start
            nx_triplets.append((h, rel_index, {'type': max_path_len}))  # relation context
            nx_triplets.append((t, rel_index, {'type': max_path_len+1}))  # relation context  这个属性时什么意思咧？
        # ？？？？？？？？？？？？？
        sig_rel_num = max_path_len + 2
        if add_reverse:  # 增加逆关系
            for i, path in enumerate(path_list):
                for kk, rel in enumerate(path):  # 0 1 2
                    rel_index = relations.index(rel) + rel_index_start
                    nx_triplets.append((rel_index, i, {'type': kk + sig_rel_num}))  # path -> relation
            for h, t, r in type_edges:
                rel_index = relations.index(r) + rel_index_start
                h = h + ent_index_start
                t = t + ent_index_start
                nx_triplets.append((rel_index, h, {'type': max_path_len + sig_rel_num}))  # relation context
                nx_triplets.append((rel_index, t, {'type': max_path_len + 1 + sig_rel_num}))  # relation context

        cg_nx.add_edges_from(nx_triplets)

        cg_dgl = dgl.DGLGraph(multigraph=True)
        cg_dgl.from_networkx(cg_nx, edge_attrs=['type'])


        # 构造特征  缺失部分使用-1代替
        f1 = np.ones((nodes_len, max_path_len), dtype=np.int) * -1  # path特征
        f2 = np.ones(nodes_len, dtype=np.int) * -1  # relation特征
        f3 = np.ones(nodes_len, dtype=np.int) * -1  # ent特征
        tar_rel = np.zeros(nodes_len, dtype=np.int)  # 是否是目标关系 目标关系index的值为1，其余为0
        # 构造路径特征
        for i, path in enumerate(path_list):
            f1[i][:len(path)] = np.array(path)
        # 构造关系特征
        for i, rel in enumerate(relations):
            f2[rel_index_start + i] = rel
            if rel == target_label:  # 目标关系index
                tar_rel[rel_index_start + i] = 1
        # 构造实体特征
        for i, ent in enumerate(entity_list):
            f3[ent_index_start + i] = ent

        index1 = np.zeros(nodes_len, dtype=np.int)
        index2 = np.zeros(nodes_len, dtype=np.int)
        index3 = np.zeros(nodes_len, dtype=np.int)
        index1[:rel_index_start] = 1
        index2[rel_index_start:ent_index_start] = 1
        index3[ent_index_start:] = 1
        # 给节点添加属性
        cg_dgl.ndata['f1'] = f1
        cg_dgl.ndata['f2'] = f2
        cg_dgl.ndata['f3'] = f3
        cg_dgl.ndata['index1'] = index1
        cg_dgl.ndata['index2'] = index2
        cg_dgl.ndata['index3'] = index3
        cg_dgl.ndata['tar_rel'] = tar_rel  # 是否是目标关系
        cg_dgl.ndata['rel_label'] = np.array([target_label]*nodes_len)  # 目标关系

        return cg_dgl

    def nei_rel_path(self, g, rel_labels, r_emb_out):
        """ Neighboring relational path module """
        # Only consider in-degree relations first.
        nei_rels = g.ndata['in_nei_rels']
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        heads_rels = nei_rels[head_ids]
        tails_rels = nei_rels[tail_ids]

        # Extract neighboring relational paths 抽取关系路径模块，可以在这里改～
        batch_paths = []
        for (head_rels, r_t, tail_rels) in zip(heads_rels, rel_labels, tails_rels):
            paths = []
            for h_r in head_rels:
                for t_r in tail_rels:
                    path = [h_r, r_t, t_r]
                    paths.append(path)
            batch_paths.append(paths)       # [B, n_paths, 3] , n_paths = n_head_rels * n_tail_rels
        # print("batch_paths",batch_paths)
        batch_paths = torch.LongTensor(batch_paths).to(rel_labels.device)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        batch_size = batch_paths.shape[0]
        batch_paths = batch_paths.view(batch_size * len(paths), -1) # [B * n_paths, 3]

        batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # Input RNN 使用rnn 模块来对路径表示进行学习
        _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        last_state = last_state.squeeze(0) # squeeze the dim 0 
        last_state = last_state.view(batch_size, len(paths), self.params.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention 使用注意力机制来汇聚路径的表示，可以在这里改～
        if self.params.path_agg == 'mean':
            output = torch.mean(last_state, 1) # [B, inp_dim]
        
        if self.params.path_agg == 'att':
            r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
            atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
            atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
            output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        else:
            raise ValueError('unknown path_agg')
        # print("out:",output.shape) out: torch.Size([64, 32])
        return output # [B, inp_dim]

    def get_logits(self, s_G, s_g_pos, s_g_cor): 
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret
    
    def forward(self, data, is_return_emb=False, cor_graph=False):
        # Initialize the embedding of entities
        g, rel_labels = data
        
        # Neighboring Relational Feature Module
        ## Initialize the embedding of nodes by neighbor relations
        if self.params.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)
        
        # Corrupt the node feature
        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]  
        
        # r: Embedding of relation
        r = self.rel_emb.weight.clone()
        
        # Input graph into GNN to get embeddings. 相比grail，多返回了关系的嵌入
        g.ndata['h'], r_emb_out = self.gnn(g, r)
        
        # GRU layer for nodes 将L层的节点喂入GRU来获得更强的表达能力
        graph_sizes = g.batch_num_nodes
        out_dim = self.params.num_gcn_layers * self.params.emb_dim
        g.ndata['repr'] = F.relu(self.batch_gru(g.ndata['repr'].view(-1, out_dim), graph_sizes))
        node_hiddens = F.relu(self.W_o(g.ndata['repr']))  # num_nodes x hidden 
        g.ndata['repr'] = self.dropout(node_hiddens)  # num_nodes x hidden
        # 子图表示
        g_out = mean_nodes(g, 'repr').view(-1, out_dim)

        # Get embedding of target nodes (i.e. head and tail nodes)
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out,
                               head_embs.view(-1, out_dim),
                               tail_embs.view(-1, out_dim),
                               F.embedding(rel_labels, r_emb_out, padding_idx=-1)], dim=1)
        else:
            g_rep = torch.cat([g_out, self.rel_emb(rel_labels)], dim=1)
        
        # Represent subgraph by composing (h,r,t) in some way. (Not use in paper)
        if self.params.comp_hrt:
            edge_embs = self.comp_hrt_emb(head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim), F.embedding(rel_labels, r_emb_out, padding_idx=-1))
            g_rep = torch.cat([g_out, edge_embs], dim=1)

        # Model neighboring relational paths 
        if self.params.nei_rel_path:
            # Model neighboring relational path
            # g_p = self.nei_rel_path(g, rel_labels, r_emb_out)
            g_p = self.get_rel_path(g, rel_labels, r_emb_out)
            g_rep = torch.cat([g_rep, g_p], dim=1)

            # g_p_ori = self.nei_rel_path(g, rel_labels, r_emb_out)
            # g_rep = torch.cat([g_rep,g_p_ori], dim=1)

            s_g = torch.cat([g_out, g_p], dim=1)
        else:
            s_g = g_out
        output = self.fc_layer(g_rep)

        self.r_emb_out = r_emb_out
        
        if not is_return_emb:
            return output
        else:
            # Get the subgraph-level embedding
            s_G = s_g.mean(0)
            return output, s_G, s_g



