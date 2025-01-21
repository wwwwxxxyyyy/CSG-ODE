from lib.base_models import VAE_Baseline
import lib.utils as utils
import torch

class LatentGraphODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
				 z0_prior, device, obsrv_std=None):

		super(LatentGraphODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.latent_dim =latent_dim



	def get_reconstruction(self, batch_feature_en,
								 batch_edges_importance_en,
                          		 batch_observed_mask_en,
                         		 batch_avg_interval_en,
                                 batch_time_pos_en,
								 batch_length_en,
								 batch_de, batch_g,
								 n_traj_samples=1,
								 run_backwards=True):

  
		first_point_mu, first_point_std = self.encoder_z0(batch_feature_en,
														  batch_edges_importance_en,
                          								  batch_observed_mask_en,
                         							      batch_avg_interval_en,
                                 						  batch_time_pos_en,
                                 						  batch_length_en) 
		means_z0 = first_point_mu.repeat(n_traj_samples,1,1) 
		sigmas_z0 = first_point_std.repeat(n_traj_samples,1,1) 
		first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0) 


		first_point_std = first_point_std.abs()
		first_point_std = torch.clamp(first_point_std, min=1e-8)
		time_steps_to_predict = batch_de["time_steps"] 



		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())
		assert (not torch.isnan(first_point_enc).any())



		
		sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)


    
		pred_x = self.decoder(sol_y)


		all_extra_info = {
			"first_point": (torch.unsqueeze(first_point_mu,0), torch.unsqueeze(first_point_std,0), first_point_enc),
			"latent_traj": sol_y.detach()
		}

		return pred_x, all_extra_info, None









