import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence
from scipy.linalg import expm
import math


class ParseData(object):

    def __init__(self, dataset_path, args, suffix='_springs5', mode="interp"):
        self.dataset_path = dataset_path  
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step  
        self.num_pre = args.extrap_num  

        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_data(self, sample_percent, batch_size, data_type="train"):
        self.batch_size = batch_size
        self.sample_percent = sample_percent
        if data_type == "train":
            cut_num = 20000
        else:  
            cut_num = 5000

  
        loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[
              :cut_num] 
        vel = np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True)[
              :cut_num] 
        edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[
                :cut_num]  
        times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[
                :cut_num] 

        self.num_graph = loc.shape[0]  
        self.num_atoms = loc.shape[1]  
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]  
        print("number graph in   " + data_type + "   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)
        print("number feature in   " + data_type + "   is %d" % self.feature)


        if self.max_loc == None:
            loc, max_loc, min_loc = self.normalize_features(loc, self.num_atoms)  

            vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
            self.max_loc = max_loc  
            self.min_loc = min_loc
            self.max_vel = max_vel
            self.min_vel = min_vel
        else:
            loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
            vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

   
        if self.mode == "interp":  
            loc_en, vel_en, times_en, times_en_original = self.interp_extrap(loc, vel, times, self.mode, data_type)
       
            loc_de = loc_en
            vel_de = vel_en
            times_de = times_en
        elif self.mode == "extrap":  
            loc_en, vel_en, times_en, loc_de, vel_de, times_de, times_en_original = self.interp_extrap(loc, vel, times,
                                                                                                       self.mode,
                                                                                                       data_type)

  
        series_list_observed, loc_observed, vel_observed, times_observed, times_observed_original = self.split_data(
            loc_en, vel_en, times_en, times_en_original)
       

        if self.mode == "interp":  
            time_begin = 0
        else:  
            time_begin = 1

        edges_importance = self.get_edges_importance(edges)  
        edges_observed_Mask, times_observed_original_total_step = self.get_edges_observed_Mask(times_observed_original,
                                                                                               self.mode, data_type)
        
        feature, features_loc, features_vel = self.get_features_60(times_observed_original, loc_observed, vel_observed,
                                                                   self.mode, data_type)
    
        edges_average_gap = self.get_edges_average_gap(times_observed_original_total_step)  
        time_pos = self.get_time_pos(times_observed_original_total_step, time_begin=time_begin) 

   
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])  
        edges = np.array((edges + 1) / 2, dtype=np.int64) 
        edges = torch.LongTensor(edges)
 
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms]) 
        edges = edges[:, off_diag_idx]  
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        feature = feature.transpose(0, 2, 1, 3) 
        times_observed_original_total_step = times_observed_original_total_step.transpose(0, 2, 1)  
        times_observed_original_total_step[times_observed_original_total_step > 0] = 1  
        length = np.amax(times_observed_original_total_step.sum(1), axis=1).reshape(self.num_graph, 1)  
 
        edges_average_gap = edges_average_gap.transpose(0, 2, 1) 
        time_pos = time_pos.transpose(0, 2, 1) 

        feature = torch.FloatTensor(feature)
        edges_importance = torch.FloatTensor(edges_importance)

        times_observed_original_total_step = torch.FloatTensor(times_observed_original_total_step)
        edges_average_gap = torch.FloatTensor(edges_average_gap)
        time_pos = torch.FloatTensor(time_pos)
        length = torch.FloatTensor(length)

     
        feature_encoder_data_loader = Loader(feature, batch_size=self.batch_size)  
        edges_importance_encoder_data_loader = Loader(edges_importance, batch_size=self.batch_size) 
        observed_mask_encoder_data_loader = Loader(times_observed_original_total_step,
                                                   batch_size=self.batch_size)  
        avg_interval_encoder_data_loader = Loader(edges_average_gap, batch_size=self.batch_size)
        time_pos_encoder_data_loader = Loader(time_pos, batch_size=self.batch_size)  
        length_encoder_data_loader = Loader(length, batch_size=self.batch_size)  


        if self.mode == "interp":
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            series_list_de = self.decoder_data(loc_de, vel_de, times_de)

        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(batch))



        num_batch = len(decoder_data_loader)
        feature_encoder_data_loader = utils.inf_generator(feature_encoder_data_loader)
        edges_importance_encoder_data_loader = utils.inf_generator(edges_importance_encoder_data_loader)
        observed_mask_encoder_data_loader = utils.inf_generator(observed_mask_encoder_data_loader)
        avg_interval_encoder_data_loader = utils.inf_generator(avg_interval_encoder_data_loader)
        time_pos_encoder_data_loader = utils.inf_generator(time_pos_encoder_data_loader)
        length_encoder_data_loader = utils.inf_generator(length_encoder_data_loader)

        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return feature_encoder_data_loader, edges_importance_encoder_data_loader, observed_mask_encoder_data_loader, avg_interval_encoder_data_loader, time_pos_encoder_data_loader, length_encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def interp_extrap(self, loc, vel, times, mode, data_type):
       
        loc_observed = np.ones_like(loc) 
        vel_observed = np.ones_like(vel)  
        times_observed = np.ones_like(times)  
        if mode == "interp":  
            if data_type == "test":
                for i in range(self.num_graph): 
                    for j in range(self.num_atoms):  
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]  
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed, vel_observed, times_observed / self.total_step, times_observed
            else: 
                return loc, vel, times / self.total_step, times


        elif mode == "extrap": 

          
            loc_observed = np.ones_like(loc)
            vel_observed = np.ones_like(vel)
            times_observed = np.ones_like(times)

            loc_extrap = np.ones_like(loc) 
            vel_extrap = np.ones_like(vel)  
            times_extrap = np.ones_like(times) 

            if data_type == "test":
                for i in range(self.num_graph):  
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                        loc_extrap[i][j] = loc[i][j][-self.num_pre:]
                        vel_extrap[i][j] = vel[i][j][-self.num_pre:]
                        times_extrap[i][j] = times[i][j][-self.num_pre:]
                times_observed_original = times_observed
                times_observed = times_observed / self.total_step  
                times_extrap = (times_extrap - self.total_step) / self.total_step
              
            else: 
             
                for i in range(self.num_graph):  
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        times_current_mask = np.where(times_current < self.total_step // 2, times_current, 0)
                        num_observe_current = np.argmax(times_current_mask) + 1
                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        vel_observed[i][j] = vel[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        vel_extrap[i][j] = vel[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:]
                times_observed_original = times_observed
                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step // 2) / self.total_step

            return loc_observed, vel_observed, times_observed, loc_extrap, vel_extrap, times_extrap, times_observed_original

    def split_data(self, loc, vel, times, times_origin):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)
        times_observed_origin = np.ones_like(times_origin)


        loc_list = []
        vel_list = []
        times_list = []
        times_origin_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:]) 
                vel_list.append(vel[i][j][1:]) 
                times_list.append(times[i][j][1:]) 
                times_origin_list.append(times_origin[i][j][1:])  
        series_list = []
 
        for i, loc_series in enumerate(loc_list):
        
            graph_index = i // self.num_atoms  
            atom_index = i % self.num_atoms  
            length = len(loc_series) 
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
       
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]
            times_observed_origin[graph_index][atom_index] = times_origin_list[i][preserved_idx]

            feature_predict = np.zeros((self.timelength, self.feature)) 
            times_predict = -1 * np.ones(self.timelength)  
            mask_predict = np.zeros((self.timelength, self.feature)) 

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list, loc_observed, vel_observed, times_observed, times_observed_origin

    def decoder_data(self, loc, vel, times):

 
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
          
            feature_predict = np.zeros((self.timelength, self.feature))  
            times_predict = -1 * np.ones(self.timelength) 
            mask_predict = np.zeros((self.timelength, self.feature))  

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list

    def variable_time_collate_fn_activity(self, batch):
        
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True)  
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])

        for b, (tt, vals, mask) in enumerate(batch):
            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

   
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:, 1:, :]
        combined_mask = combined_mask[:, 1:, :]

        combined_tt = combined_tt.float()

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
        }
        return data_dict

    def normalize_features(self, inputs, num_balls):
        
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]
  
        self.timelength = max(value_list_length)  
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
       
        value_padding = pad_sequence(value_list, batch_first=True, padding_value=0)
        max_value = torch.max(value_padding).item()  
        min_value = torch.min(value_padding).item()  

        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs, max_value, min_value

    def convert_sparse(self, graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr
    

    def exp_0(self, edge):
      
        return expm(edge) - np.eye(edge.shape[0])

    def frobenius_norm(self, L_f):
       
        return np.linalg.norm(L_f, 'fro')

    def get_edges_importance(self, edge):
       
        A = np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)
        edge = edge * A
        L_f = np.zeros_like(edge, dtype=np.float32)  
        Y_L_f = np.zeros_like(edge, dtype=np.float32) 
        h = (2e-4) / self.num_atoms
        uvT = np.ones([self.num_atoms, self.num_atoms])
        edge_T = edge.transpose(0, 2, 1)  
        total_tranmisson = []
        for i in range(self.num_graph):
            f1 = self.exp_0(edge_T[i] + h * uvT)
            f2 = self.exp_0(edge_T[i] - h * uvT)
            L_f[i] = (f1 - f2) / (h * 2)
            total_tranmisson.append(self.frobenius_norm(L_f[i]))
            Y_L_f[i] = (edge[i] * L_f[i]) / total_tranmisson[i]

        return np.array(Y_L_f)

    def get_edges_observed_Mask(self, times_observed, mode, data_type):

        if mode == "interp":
            total_step = 60
        elif mode == "extrap":
            if data_type == "train":
                total_step = 30
            elif data_type == "test":
                total_step = 60
        times = np.zeros([self.num_graph, self.num_atoms, total_step], dtype=int) 
        edges_mask = np.zeros([self.num_graph, total_step, self.num_atoms, self.num_atoms], dtype=int)

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                times[i][j][times_observed[i][j]] = 1
        times = times.transpose(0, 2, 1) 
        for i in range(self.num_graph):
            for j in range(total_step):
                edges_mask[i][j] = times[i][j].reshape(-1, 1) @ times[i][j].reshape(1, -1)
        return edges_mask, times.transpose(0, 2, 1)  

    def get_edges_average_gap(self, times_observed_original_total_step):

        total_step = times_observed_original_total_step.shape[2]
        edges_average_gap = np.ones([self.num_graph, self.num_atoms, total_step]) * (total_step / 2)
        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                for k in range(total_step):
                    if k == 0:
                        if times_observed_original_total_step[i][j][k + 1] != 0:
                            edges_average_gap[i][j][k] = (k + 1 - k)
                    if k == total_step - 1:
                        if times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k - k - 1)
                    else:
                        if times_observed_original_total_step[i][j][k + 1] != 0 and \
                                times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k + 1 - k - 1) / 2
                        elif times_observed_original_total_step[i][j][k + 1] != 0 and \
                                times_observed_original_total_step[i][j][k - 1] == 0:
                            edges_average_gap[i][j][k] = (k + 1 - k)
                        elif times_observed_original_total_step[i][j][k + 1] == 0 and \
                                times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k - k - 1)

        return edges_average_gap  

    def get_features_60(self, times, loc, vel, mode, data_type):
      

        if mode == "interp":
            total_step = 60
        elif mode == "extrap":
            if data_type == "train":
                total_step = 30
            elif data_type == "test":
                total_step = 60
      
        features_loc = np.zeros([self.num_graph, self.num_atoms, total_step, 2]) 
        features_vel = np.zeros([self.num_graph, self.num_atoms, total_step, 2]) 
        for i in range(times.shape[0]):
            for j in range(times.shape[1]):
                features_loc[i][j][times[i][j]] = loc[i][j]
                features_vel[i][j][times[i][j]] = vel[i][j]
        features = np.concatenate([features_loc, features_vel], axis=-1)
        return features, features_loc, features_vel  

    def get_time_pos(self, times, time_begin=0):
        total_step = times.shape[2]
        indices = torch.arange(total_step)
        times_pos = times
        for i in range(times.shape[0]):
            for j in range(times.shape[1]):
                for k in range(times.shape[2]):
                    if times[i][j][k] != 0:
                        times_pos[i][j][k] = indices[k]
        times_pos = times_pos - time_begin
        return times_pos / total_step  


class ParseData_motion(object):

    def __init__(self, dataset_path, args, suffix='_motion_walk', mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.num_pre = args.extrap_num

        self.max_loc = None
        self.min_loc = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_data(self, sample_percent, batch_size, data_type="train"):
        self.batch_size = batch_size
        self.sample_percent = sample_percent
        if self.suffix == "_motion_walk":
            if data_type == "train":
                cut_num = 125
            else:
                cut_num = 27
        elif self.suffix == "_PEMS08":
            if data_type == "train":
                cut_num = 199
            else:
                cut_num = 49
        elif self.suffix == "_motion_jump":
            if data_type == "train":
                cut_num = 257
            else:
                cut_num = 55

        loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0]
        print("number graph in   " + data_type + "   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)
        print("number feature in   " + data_type + "   is %d" % self.feature)

        if self.max_loc == None:
            loc, max_loc, min_loc = self.normalize_features(loc, self.num_atoms)
            self.max_loc = max_loc
            self.min_loc = min_loc
        else:
            loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1

        if self.mode == "interp":
            loc_en, times_en, times_en_original = self.interp_extrap(loc, times, self.mode, data_type)
            loc_de = loc_en
            times_de = times_en
        elif self.mode == "extrap":
            loc_en, times_en, loc_de, times_de, times_en_original = self.interp_extrap(loc, times, self.mode,
                                                                                       data_type)


        series_list_observed, loc_observed, times_observed, times_observed_original = self.split_data(
            loc_en, times_en, times_en_original)

        if self.mode == "interp":
            time_begin = 0
        else:
            time_begin = 1

        edges_importance = self.get_edges_importance(edges)
        edges_observed_Mask, times_observed_original_total_step = self.get_edges_observed_Mask(times_observed_original,
                                                                                               self.mode, data_type)
        feature, features_loc = self.get_features_60(times_observed_original, loc_observed, self.mode, data_type)
        edges_average_gap = self.get_edges_average_gap(times_observed_original_total_step)
        time_pos = self.get_time_pos(times_observed_original_total_step, time_begin=time_begin)


        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)

        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])
        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        feature = feature.transpose(0, 2, 1, 3)
        times_observed_original_total_step = times_observed_original_total_step.transpose(0, 2, 1)
        times_observed_original_total_step[times_observed_original_total_step > 0] = 1
        length = np.amax(times_observed_original_total_step.sum(1), axis=1).reshape(self.num_graph, 1)
        edges_average_gap = edges_average_gap.transpose(0, 2, 1)
        time_pos = time_pos.transpose(0, 2, 1)

        feature = torch.FloatTensor(feature)
        edges_importance = torch.FloatTensor(edges_importance)
        times_observed_original_total_step = torch.FloatTensor(times_observed_original_total_step)
        edges_average_gap = torch.FloatTensor(edges_average_gap)
        time_pos = torch.FloatTensor(time_pos)
        length = torch.FloatTensor(length)


        feature_encoder_data_loader = Loader(feature, batch_size=self.batch_size)
        edges_importance_encoder_data_loader = Loader(edges_importance, batch_size=self.batch_size)
        observed_mask_encoder_data_loader = Loader(times_observed_original_total_step, batch_size=self.batch_size)
        avg_interval_encoder_data_loader = Loader(edges_average_gap, batch_size=self.batch_size)
        time_pos_encoder_data_loader = Loader(time_pos, batch_size=self.batch_size)
        length_encoder_data_loader = Loader(length, batch_size=self.batch_size)


        if self.mode == "interp":
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            series_list_de = self.decoder_data(loc_de, times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(batch))

        num_batch = len(decoder_data_loader)
        feature_encoder_data_loader = utils.inf_generator(feature_encoder_data_loader)
        edges_importance_encoder_data_loader = utils.inf_generator(edges_importance_encoder_data_loader)
        observed_mask_encoder_data_loader = utils.inf_generator(observed_mask_encoder_data_loader)
        avg_interval_encoder_data_loader = utils.inf_generator(avg_interval_encoder_data_loader)
        time_pos_encoder_data_loader = utils.inf_generator(time_pos_encoder_data_loader)
        length_encoder_data_loader = utils.inf_generator(length_encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return feature_encoder_data_loader, edges_importance_encoder_data_loader, observed_mask_encoder_data_loader, avg_interval_encoder_data_loader, time_pos_encoder_data_loader, length_encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def interp_extrap(self, loc, times, mode, data_type):
  
        loc_observed = np.ones_like(loc)
        times_observed = np.ones_like(times)
        if mode == "interp":
            if data_type == "test":
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed, times_observed / self.total_step, times_observed

            else:
                return loc, times / self.total_step, times


        elif mode == "extrap":


            loc_observed = np.ones_like(loc)
            times_observed = np.ones_like(times)

            loc_extrap = np.ones_like(loc)
            times_extrap = np.ones_like(times)

            if data_type == "test":
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                        loc_extrap[i][j] = loc[i][j][-self.num_pre:]
                        times_extrap[i][j] = times[i][j][-self.num_pre:]
                times_observed_original = times_observed
                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step) / self.total_step
            else:
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        times_current_mask = np.where(times_current < self.total_step // 2, times_current, 0)
                        num_observe_current = np.argmax(times_current_mask) + 1

                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:]
                times_observed_original = times_observed
                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step // 2) / self.total_step

            return loc_observed, times_observed, loc_extrap, times_extrap, times_observed_original

    def split_data(self, loc, times, times_origin):
        loc_observed = np.ones_like(loc)
        times_observed = np.ones_like(times)
        times_observed_origin = np.ones_like(times_origin)
        loc_list = []
        times_list = []
        times_origin_list = []

        for i in range(self.num_graph): 
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  
                times_list.append(times[i][j][1:]) 
                times_origin_list.append(times_origin[i][j][1:])  
        series_list = []

        for i, loc_series in enumerate(loc_list):

            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))

            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]
            times_observed_origin[graph_index][atom_index] = times_origin_list[i][preserved_idx]


            feature_predict = np.zeros((self.timelength, self.feature))
            times_predict = -1 * np.ones(self.timelength)
            mask_predict = np.zeros((self.timelength, self.feature))

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = loc_series
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list, loc_observed, times_observed, times_observed_origin

    def decoder_data(self, loc, times):


        loc_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):

            feature_predict = np.zeros((self.timelength, self.feature))
            times_predict = -1 * np.ones(self.timelength)
            mask_predict = np.zeros((self.timelength, self.feature))

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = loc_series
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list

    def variable_time_collate_fn_activity(self, batch):

        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True)  
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])

        for b, (tt, vals, mask) in enumerate(batch):
            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask


        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:, 1:, :]
        combined_mask = combined_mask[:, 1:, :]

        combined_tt = combined_tt.float()

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
        }
        return data_dict

    def normalize_features(self, inputs, num_balls):

        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]

        value_padding = pad_sequence(value_list, batch_first=True, padding_value=0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs, max_value, min_value

    def convert_sparse(self, graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr

    def exp_0(self, edge):
        return expm(edge) - np.eye(edge.shape[0])

    def frobenius_norm(self, L_f):
        return np.linalg.norm(L_f, 'fro')

    def get_edges_importance(self, edge):

        A = np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)
        edge = edge * A
        L_f = np.zeros_like(edge, dtype=np.float32)
        Y_L_f = np.zeros_like(edge, dtype=np.float32)
        h = (2e-4) / self.num_atoms
        uvT = np.ones([self.num_atoms, self.num_atoms])
        edge_T = edge.transpose(0, 2, 1)
        total_tranmisson = []
        for i in range(self.num_graph):
            f1 = self.exp_0(edge_T[i] + h * uvT)
            f2 = self.exp_0(edge_T[i] - h * uvT)
            L_f[i] = (f1 - f2) / (h * 2)
            total_tranmisson.append(self.frobenius_norm(L_f[i]))
            Y_L_f[i] = (edge[i] * L_f[i]) / total_tranmisson[i]
        return np.array(Y_L_f)

    def get_edges_observed_Mask(self, times_observed, mode, data_type):

        if self.suffix == "_motion_walk" or self.suffix == "_motion_jump":
            if mode == "interp":
                total_step = 50
            elif mode == "extrap":
                if data_type == "train":
                    total_step = 25
                elif data_type == "test":
                    total_step = 50
        elif self.suffix == "_PEMS08":
            if mode == "interp":
                total_step = 60
            elif mode == "extrap":
                if data_type == "train":
                    total_step = 30
                elif data_type == "test":
                    total_step = 60
        times = np.zeros([self.num_graph, self.num_atoms, total_step], dtype=int)
        edges_mask = np.zeros([self.num_graph, total_step, self.num_atoms, self.num_atoms], dtype=int)

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                times[i][j][times_observed[i][j]] = 1
        times = times.transpose(0, 2, 1)
        for i in range(self.num_graph):
            for j in range(total_step):
                edges_mask[i][j] = times[i][j].reshape(-1, 1) @ times[i][j].reshape(1, -1)
        return edges_mask, times.transpose(0, 2, 1)

    def get_edges_average_gap(self, times_observed_original_total_step):

        total_step = times_observed_original_total_step.shape[2]
        edges_average_gap = np.ones([self.num_graph, self.num_atoms, total_step]) * (total_step / 2)
        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                for k in range(total_step):
                    if k == 0:
                        if times_observed_original_total_step[i][j][k + 1] != 0:
                            edges_average_gap[i][j][k] = (k + 1 - k)
                    if k == total_step - 1:
                        if times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k - k - 1)
                    else:
                        if times_observed_original_total_step[i][j][k + 1] != 0 and \
                                times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k + 1 - k - 1) / 2
                        elif times_observed_original_total_step[i][j][k + 1] != 0 and \
                                times_observed_original_total_step[i][j][k - 1] == 0:
                            edges_average_gap[i][j][k] = (k + 1 - k)
                        elif times_observed_original_total_step[i][j][k + 1] == 0 and \
                                times_observed_original_total_step[i][j][k - 1] != 0:
                            edges_average_gap[i][j][k] = (k - k - 1)

        return edges_average_gap  

    def get_features_60(self, times, loc, mode, data_type):
        if self.suffix == "_motion_walk" or self.suffix == "_motion_jump":
            if mode == "interp":
                total_step = 50
            elif mode == "extrap":
                if data_type == "train":
                    total_step = 25
                elif data_type == "test":
                    total_step = 50
        elif self.suffix == "_PEMS08":
            if mode == "interp":
                total_step = 60
            elif mode == "extrap":
                if data_type == "train":
                    total_step = 30
                elif data_type == "test":
                    total_step = 60
        if self.suffix == "_motion_walk" or self.suffix == "_motion_jump":
            features_loc = np.zeros([self.num_graph, self.num_atoms, total_step, 6])  
        elif  self.suffix == "_PEMS08":
            features_loc = np.zeros([self.num_graph, self.num_atoms, total_step, 3])

        for i in range(times.shape[0]):
            for j in range(times.shape[1]):
                features_loc[i][j][times[i][j]] = loc[i][j]
        features = features_loc
        return features, features_loc

    def get_time_pos(self, times, time_begin=0):
        total_step = times.shape[2]
        indices = torch.arange(total_step)
        times_pos = times
        for i in range(times.shape[0]):
            for j in range(times.shape[1]):
                for k in range(times.shape[2]):
                    if times[i][j][k] != 0:
                        times_pos[i][j][k] = indices[k]
        times_pos = times_pos - time_begin
        return times_pos / total_step