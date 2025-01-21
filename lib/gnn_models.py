import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
import math
import lib.utils as utils
from einops import *





class NRIConv(nn.Module):
    """MLP decoder module."""

    def __init__(self, in_channels, out_channels, dropout=0., skip_first=False):
        super(NRIConv, self).__init__()

        self.edge_types = 2
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_out_shape = out_channels
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(in_channels + out_channels, out_channels)
        self.out_fc2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.rel_type = None
        self.rel_rec = None
        self.rel_send = None

    def forward(self, inputs, pred_steps=1):
   
        rel_type = self.rel_type 
        rel_rec = self.rel_rec
        rel_send = self.rel_send

      
        receivers = torch.matmul(rel_rec, inputs)  
        senders = torch.matmul(rel_send, inputs)  
        pre_msg = torch.cat([senders, receivers], dim=-1) 

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape) 

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

     
        for i in range(start_idx, len(self.msg_fc2)): 
            msg = F.relu(self.msg_fc1[i](pre_msg)) 
            msg = self.dropout(msg)
            msg = F.relu(self.msg_fc2[i](msg)) 
            msg = msg * rel_type[:, :, i:i + 1]  
            all_msgs += msg

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1) 


        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)


        pred = self.dropout(F.relu(self.out_fc1(aug_inputs)))
        pred = self.dropout(F.relu(self.out_fc2(pred)))


        return inputs + pred


class NRI(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layers, dropout=0.2):
        super(NRI, self).__init__()
        self.gcs = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(in_dim, n_hid)
        self.out_w_ode = nn.Linear(n_hid, out_dim)

        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.out_w_ode)

        self.layer_norm = nn.LayerNorm(n_hid)
        for l in range(n_layers):
            self.gcs.append(NRIConv(n_hid, n_hid, dropout))

    def forward(self, x): 
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        for gc in self.gcs:  
            h_t = gc(h_t)  
        h_out = self.out_w_ode(h_t)

        return h_out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        utils.init_network_weights(self.layers)

    def forward(self, x):
        return self.layers(x)


class Linear_Param(nn.Module):
    def __init__(self, input_size, output_size, query_vector_dim):
        super(Linear_Param, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, input_size, output_size), requires_grad=True)
        self.b_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, output_size), requires_grad=True)

        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.b_1)

    def forward(self, x, query_vectors):
        W_1 = torch.einsum("nd, dio->nio", query_vectors, self.W_1)
        b_1 = torch.einsum("nd, do->no", query_vectors, self.b_1)
        x = torch.squeeze(torch.bmm(x.unsqueeze(1), W_1)) + b_1
        return x


class AGCRNCellWithLinear(nn.Module):
    def __init__(self, input_size, query_vector_dim):
        super(AGCRNCellWithLinear, self).__init__()
        self.update_gate = Linear_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.reset_gate = Linear_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.candidate_gate = Linear_Param(2 * input_size + 1, input_size, query_vector_dim)

    def forward(self, x, h, query_vectors, adj, nodes_ind):

        combined = torch.cat([x, h], dim=-1)  # [256,5,129]
        combined = torch.matmul(adj, combined)  # [256,5,129]
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], query_vectors)) 
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], query_vectors))  

        h[nodes_ind] = r * h[nodes_ind]
        combined_new = torch.cat([x, h], dim=-1)
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], query_vectors))
      
        return (1 - u) * h[nodes_ind] + u * candidate_h  


class VSDGCRNN(nn.Module):
    def __init__(self, input_dim, d_model, num_of_nodes, batch_size, rarity_alpha=0.5, query_vector_dim=32):
        super(VSDGCRNN, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model  
        self.num_of_nodes = num_of_nodes
        self.batch_size = batch_size

        self.gated_update = AGCRNCellWithLinear(d_model, query_vector_dim)
        self.rarity_alpha = rarity_alpha  
        self.rarity_W = nn.Parameter(torch.empty(num_of_nodes, num_of_nodes), requires_grad=True) 
        self.rarity_W_edge_importance = nn.Parameter(torch.empty(num_of_nodes, num_of_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.rarity_W_edge_importance)
        nn.init.xavier_uniform_(self.rarity_W)
        self.relu = nn.ReLU()
        self.adj = nn.Parameter(torch.empty(batch_size, self.num_of_nodes, self.num_of_nodes),
                                requires_grad=True)  
        nn.init.xavier_uniform_(self.adj)
        self.projection_f = MLP(input_dim, 2 * d_model, query_vector_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, observed_mask, avg_interval, edge_importance,length):
   
        cur_batch = edge_importance.shape[0]
        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(self.num_of_nodes).to(device), 'v x -> b v x',
                   b=cur_batch)  
        output = torch.zeros_like(h)  
        nodes_initial_mask = torch.zeros(cur_batch, self.num_of_nodes).to(device)  

        var_total_obs = torch.sum(observed_mask, dim=1) 
        var_total_obs_ = repeat(var_total_obs, 'b v -> b v x', x=self.input_dim) 

        query_vectors = self.projection_f(
            obs_emb.sum(1) / var_total_obs.unsqueeze(-1))  
        adj = torch.softmax(self.relu(self.adj[:cur_batch]), dim=-1)  

        adj_add_importance = adj + self.rarity_W_edge_importance * edge_importance  
       
        for step in range(int(torch.max(length).item())):  
            adj_mask = torch.zeros(size=[cur_batch, self.num_of_nodes, self.num_of_nodes]).to(device)
            cur_obs = obs_emb[:, step]  
            cur_mask = observed_mask[:, step]  
            cur_obs_var = torch.where(cur_mask)  
            nodes_initial_mask[cur_obs_var] = 1  
            nodes_need_update = cur_obs_var  
            cur_avg_interval = avg_interval[:, step]  

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1)) 
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=self.num_of_nodes) 
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=self.num_of_nodes)  
            rarity_score_matrix = -1 * self.rarity_W * (
                torch.abs(rarity_score_matrix_row - rarity_score_matrix_col)) 

            if nodes_need_update[0].shape[0] > 0:
                
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), self.num_of_nodes).to(
                    device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), self.num_of_nodes).to(device)
               
                cur_adj = adj_add_importance * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I 
                

                h[nodes_need_update] = self.gated_update(torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                                                         h, 
                                                         query_vectors[nodes_need_update],  
                                                         cur_adj, 
                                                         nodes_need_update)

            end_sample_ind = torch.where(step == (length.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(length).item()) - 1:
                return output
        return output


class GNN(nn.Module):

    def __init__(self, in_dim, n_hid, out_dim, n_layers, num_nodes, batch_size, dropout=0.2, 
                 query_vector_dim=32, rarity_alpha=0.5):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(in_dim, n_hid)
        self.sequence_w = nn.Linear(n_hid, n_hid) 
        self.out_w_encoder = nn.Linear(n_hid, out_dim * 2)
        utils.init_network_weights(self.adapt_ws)
  

        utils.init_network_weights(self.sequence_w)
        utils.init_network_weights(self.out_w_encoder)


        self.layer_norm = nn.LayerNorm(n_hid)
        for l in range(n_layers):
            self.gcs.append(VSDGCRNN(n_hid, n_hid, num_nodes, batch_size, rarity_alpha, query_vector_dim))

   
        self.w_transfer = nn.Linear(self.n_hid + 1, self.n_hid, bias=True)  
        utils.init_network_weights(self.w_transfer)

    def forward(self, x,  
                edges_importance,  
                observed_mask, 
                avg_interval,  
                time_pos,
                length): 

        h_0 = F.relu(self.adapt_ws(x))  
        h_t = self.drop(h_0)  
        h_t = self.layer_norm(h_t)  

        for gc in self.gcs: 
            h_t = gc(h_t, observed_mask, avg_interval, edges_importance,length)
        h_ball = h_t.view(-1,self.n_hid)
       

        h_out = self.out_w_encoder(h_ball)  
        mean, mu = self.split_mean_mu(h_out) 
        mu = mu.abs()
        return mean, mu

    def split_mean_mu(self, h):
        last_dim = h.size()[-1] // 2
        res = h[:, :last_dim], h[:, last_dim:]
        return res
