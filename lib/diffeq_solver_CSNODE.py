import torch
import torch.nn as nn
import lib.utils as utils


class DynamicODESolver:
    def __init__(self, func, x0, gfunc=None, ufunc=None, u0=None, step_size=None, interp=None, atol=1e-6, norm=None):
       
        self.func = func
        self.x0 = x0
        self.gfunc = gfunc
        self.ufunc = ufunc
        self.u0 = u0 if u0 is not None else x0
        self.step_size = step_size
        self.interp = interp
        self.atol = atol  
        self.norm = norm  

    def _before_integrate(self, t): 
        pass

    def _advance(self, next_t):
        t0 = self.t
        x0 = self.x  
        u0 = self.u  
        dt = next_t - t0  
        if self.ufunc is None:
            u1 = u0
            udot = u1
        else:  
            udot = self.ufunc(t0, u0)  
            u1 = udot * dt + u0  

        if self.gfunc is None:
            gu1 = udot
        else:  
            gu1 = self.gfunc(t0, udot)  

        dx = self._step_func(t0, dt, next_t, x0, gu1)  
       

        x1 = x0 + dx  

        if self.interp == "linear":
            x_next = self._linear_interp(t0, next_t, x0, x1, next_t) 
            u_next = self._linear_interp(t0, next_t, u0, u1, next_t) 
        else:
            x_next = x1
            u_next = u1
        self.t = next_t
        self.x = x_next
        self.u = u_next
        return x_next 

    def integrate(self, t): 
        if self.step_size is None:
            self.step_size = t[1] - t[0]

        self.t = t[0]
        self.x = self.x0  
        self.u = self.u0  

        solution = torch.empty(len(t), *self.x0.shape, dtype=self.x0.dtype, device=self.x0.device)  
        solution[0] = self.x0
        t = t.to(self.x0.device, self.x0.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution

    def _step_func(self, t0, dt, t1, x0, gu1):
    
        f0 = self.func(t0, x0) + gu1
        dx = f0 * dt
        return dx

    def _linear_interp(self, t0, t1, y0, y1, t):  
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


class GraphODEFunc(nn.Module):
    def __init__(self, ode_func_net, device=torch.device("cpu")):
     
        super(GraphODEFunc, self).__init__()

        self.device = device
        self.ode_func_net = ode_func_net  
        self.nfe = 0

    def forward(self, t_local, z, backwards=False):
      
        self.nfe += 1
        grad = self.ode_func_net(z)

        if backwards:
            grad = -grad
        return grad

    def set_graph(self, rec_type, rel_rec, rel_send, edge_types):
        for layer in self.ode_func_net.gcs:
            layer.rel_type = rec_type
            layer.rel_rec = rel_rec
            layer.rel_send = rel_send
            layer.edge_types = edge_types
        self.nfe = 0



class ODEFunc(nn.Module):  
    def __init__(self, hidden_dim, num_layers, input_dim, M, num_atoms, device):
        super(ODEFunc, self).__init__()
        self.M = M

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, input_dim))
        layers.append(nn.Tanh())
        layers_M = []
        for i in range(M):
            layers_M.append(nn.Sequential(*layers).to(device))
            utils.init_network_weights(layers_M[i])
        self.net = layers_M

        A_j = [nn.Parameter(torch.empty(num_atoms, input_dim), requires_grad=True).to(device)]  
        for _ in range(M):
            A_j.append(nn.Parameter(torch.empty(num_atoms, input_dim), requires_grad=True).to(device))
        for i in range(M + 1):
            nn.init.normal_(A_j[i], 0, 0.1)
        self.A_M = A_j
        
        self.device = device

    def forward(self, t, y):
        y_result = torch.zeros_like(y).to(self.device)

        for i in range(self.M):
            y_M = self.net[i](y) * self.A_M[i + 1].unsqueeze(0)  
            y_result += y_M 
        y_result += self.A_M[0].unsqueeze(0) * y
        return y_result  

class ODEFuncg(nn.Module):
    def __init__(self, input_dim, device):
        super(ODEFuncg, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.device = device

    def forward(self, t, y):
        y_result = self.linear(y)
        return y_result