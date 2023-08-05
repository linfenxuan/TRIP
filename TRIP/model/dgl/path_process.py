import torch
import numpy as np
import networkx as nx
from .ori_path import all_simple_edge_paths

def path_generate(subgraph, rel_labels, r_emb_out, max_paths = 200, max_path_len = 3, add_reverse=True):
    nodes = np.array(subgraph.nodes())
    edges = subgraph.edges() 

    head_index = (subgraph.ndata['id'] == 1).nonzero().squeeze(1)
    tail_index = (subgraph.ndata['id'] == 2).nonzero().squeeze(1)

    edges = [np.array(x) for x in list(edges)]
    edge_type = np.array(subgraph.edata['type'].cpu())
    target_label = subgraph.edata['label'][0]

    type_edges = list(zip(edges[0], edges[1], edge_type))

    type_edges = [tuple(item) for item in type_edges]
    temp_nx = nx.MultiDiGraph(type_edges)

    batch_paths = []
    batch_neg_paths = []
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
        neg_path_list = path_list
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
