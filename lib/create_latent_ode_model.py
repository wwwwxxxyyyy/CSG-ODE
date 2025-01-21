from lib.gnn_models import GNN,NRI
from lib.latent_ode import LatentGraphODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver
from lib.diffeq_solver_CSNODE import ODEFunc, ODEFuncg, GraphODEFunc



def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):



	latent_dim = args.latents 
	rec_dim = args.rec_dims
	input_dim = input_dim
	ode_dim = args.ode_dims 
	sub_networks_dim = args.sub_networks_dims 



	encoder_z0 = GNN(in_dim=input_dim, 
				    n_hid=rec_dim, 
					out_dim=latent_dim, 
					n_layers=args.rec_layers, 
					num_nodes=args.n_balls, 
					batch_size=args.batch_size, 
					dropout=args.dropout, 
					query_vector_dim=args.query_vector_dim,
					rarity_alpha=args.rarity_alpha)  
	


	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim 
	else:
		ode_input_dim = latent_dim 

	
	ode_func_u_net = NRI(in_dim = ode_input_dim, 
					  n_hid =ode_dim, 
					  out_dim = ode_input_dim, 
					  n_layers=args.NRI_layers, 
					  dropout=args.dropout) 
	gen_ode_func_u = GraphODEFunc(ode_func_net=ode_func_u_net,device=device).to(device)


	ode_func = ODEFunc(hidden_dim = sub_networks_dim, 
					   num_layers=args.ode_func_layers, 
					   input_dim=ode_input_dim, 
					   M=args.M, 
					   num_atoms=args.n_balls, 
					   device=device)
	ode_func_g = ODEFuncg(input_dim=ode_input_dim, 
					   	  device=device)

	diffeq_solver = DiffeqSolver(ode_func=ode_func, ode_func_g=ode_func_g, ode_func_u=gen_ode_func_u, args=args, odeint_atol=1e-6, device=device)


	decoder = Decoder(latent_dim, 
				   	  input_dim).to(device) 


	model = LatentGraphODE(
		input_dim = input_dim, 
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0,  
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		).to(device)

	return model
