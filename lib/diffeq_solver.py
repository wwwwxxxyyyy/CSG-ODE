import torch
import torch.nn as nn
import numpy as np
from lib.diffeq_solver_CSNODE import DynamicODESolver as odeint_csnode


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, ode_func_g, ode_func_u, args, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()


        self.device = device 
        self.ode_func = ode_func
        self.ode_func_g = ode_func_g
        self.ode_func_u = ode_func_u 
        self.args = args
        self.num_atoms = args.n_balls 



        self.odeint_atol = odeint_atol


        self.rel_rec, self.rel_send = self.compute_rec_send()


    def compute_rec_send(self): 
        off_diag = np.ones([self.num_atoms, self.num_atoms]) - np.eye(self.num_atoms) 
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]), dtype=np.float32) 
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32) 
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        rel_send = torch.FloatTensor(rel_send).to(self.device)

        return rel_rec, rel_send


    def forward(self, first_point, time_steps_to_predict, graph, backwards = False):
       
        ispadding = False
        if time_steps_to_predict[0] != 0:
            ispadding = True
            time_steps_to_predict = torch.cat((torch.zeros(1,device=time_steps_to_predict.device),time_steps_to_predict))



        n_traj_samples, n_traj,feature = first_point.size()[0], first_point.size()[1],first_point.size()[2]
        first_point_augumented = first_point.view(-1,self.num_atoms,feature) 
        if self.args.augment_dim > 0:
            aug = torch.zeros(first_point_augumented.shape[0],first_point_augumented.shape[1], self.args.augment_dim).to(self.device)
            first_point_augumented = torch.cat([first_point_augumented, aug], 2) 
            feature += self.args.augment_dim 
        graph_augmented = torch.cat([graph for _ in range(n_traj_samples)], dim=0)

        rel_type_onehot = torch.FloatTensor(first_point_augumented.size(0), self.rel_rec.size(0),self.args.edge_types).to(self.device)  
        rel_type_onehot.zero_() 
        rel_type_onehot.scatter_(2, graph_augmented.view(first_point_augumented.size(0), -1, 1), 1)  

        self.ode_func_u.set_graph(rel_type_onehot,self.rel_rec,self.rel_send,self.args.edge_types)

       

        pred_y = odeint_csnode(func=self.ode_func,
                               x0=first_point_augumented, 
                               gfunc=self.ode_func_g, 
                               ufunc=self.ode_func_u, 
                               atol= self.odeint_atol).integrate(time_steps_to_predict)
       

        if ispadding:
            pred_y = pred_y[1:,:,:,:]
            time_steps_to_predict = time_steps_to_predict[1:]

        pred_y = pred_y.view(time_steps_to_predict.size(0), -1, pred_y.size(3)) 
        pred_y = pred_y.permute(1,0,2)
        pred_y = pred_y.view(n_traj_samples,n_traj,-1,feature) 
    
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :, :-self.args.augment_dim]

        return pred_y

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot


















