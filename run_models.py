import os
import sys
from lib.new_dataLoader import ParseData, ParseData_motion
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_latent_ode_model import create_LatentODE_model
from lib.utils import compute_loss_all_batches


parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=29,help='Number of objects in the dataset. spring/charged-5,motion-29,PEMS08-170')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=2024, help="Random_seed")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--extrap', type=str, default="False",help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')

parser.add_argument('--data', type=str, default='PEMS08', help="spring,charged,motion_walk,motion_jump,PEMS08")
parser.add_argument('--sample-percent-train', type=float, default=0.4, help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=0.4, help='Percentage of testing observtaion data')

parser.add_argument('--query_vector_dim', type=int, default=32, help="query_vector_dim")
parser.add_argument('--rarity_alpha', type=float, default=0.5, help="rarity_alpha")
parser.add_argument('--rec-dims', type=int, default=64, help="Dimensionality of the recognition model .")
parser.add_argument('--rec-layers', type=int, default=1,help="Number of layers in recognition model ") 

parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--NRI-layers', type=int, default=1,help="Number of layers  ODE_GNN func ")  

parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')

parser.add_argument('--ode-func-layers', type=int, default=1, help="subnetworks Dipth")
parser.add_argument('--M', type=int, default=2, help="Number of Subnetwork ") 
parser.add_argument('--sub-networks-dims', type=int, default=128, help="subnetworks width ") 


parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--alias', type=str, default="run")

args = parser.parse_args()


if args.data == "spring":
    args.dataset = 'data/springs5'
    args.suffix = '_springs5'
    args.total_ode_step = 60
elif args.data == "charged":
    args.dataset = 'data/charged5'
    args.suffix = '_charged5'
    args.total_ode_step = 60
elif args.data == "motion_walk":
    args.dataset = 'data/motion_walk35'
    args.suffix = '_motion_walk'
    args.total_ode_step = 49
elif args.data == "motion_jump":
    args.dataset = 'data/motion_jump118'
    args.suffix = '_motion_jump'
    args.total_ode_step = 49
elif args.data == "PEMS08":
    args.dataset = 'data/PEMS08'
    args.suffix = '_PEMS08'
    args.total_ode_step = 60


if torch.cuda.is_available():
    print("Using GPU" + "-" * 80)
    device = torch.device("cuda:0")
else:
    print("Using CPU" + "-" * 80)
    device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-" * 80)
    args.mode = "extrap"
elif args.extrap == "False":
    print("Running interp mode" + "-" * 80)
    args.mode = "interp"

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
 
        experimentID = int(SystemRandom().random() * 100000)


    print("Loading dataset: " + args.dataset)

    if args.data == "spring" or args.data == "charged":
        dataloader = ParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)
    elif args.data == "motion_walk" or args.data == "PEMS08" or args.data == "motion_jump":
        dataloader = ParseData_motion(args.dataset, suffix=args.suffix, mode=args.mode, args=args)
    test_feature_encoder, test_edges_importance_encoder, test_observed_mask_encoder, test_avg_interval_encoder, test_time_pos_encoder, test_length_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="test")
    print("Test data successfully read")
    train_feature_encoder, train_edges_importance_encoder, train_observed_mask_encoder, train_avg_interval_encoder, train_time_pos_encoder, train_length_encoder, train_decoder, train_graph, train_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train, batch_size=args.batch_size, data_type="train")
    print("Train data successfully read")

    input_dim = dataloader.feature

  
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)



    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)


    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)


    log_path = "logs/" + args.alias + "_" + args.data + "_" + str(
        args.sample_percent_train) + "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)


    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    wait_until_kl_inc = 10
    best_test_mse = np.inf
    n_iters_to_viz = 1


    def train_single_batch(model,
                           batch_dict_feature_encoder,
                           batch_dict_edges_importance_encoder,
                           batch_dict_observed_mask_encoder,
                           batch_dict_avg_interval_encoder,
                           batch_dict_time_pos_encoder,
                           batch_dict_length_encoder,
                           batch_dict_decoder,
                           batch_dict_graph,
                           kl_coef):

        optimizer.zero_grad()

        train_res = model.compute_all_losses(batch_dict_feature_encoder,
                                             batch_dict_edges_importance_encoder,
                                             batch_dict_observed_mask_encoder,
                                             batch_dict_avg_interval_encoder,
                                             batch_dict_time_pos_encoder,
                                             batch_dict_length_encoder,
                                             batch_dict_decoder,
                                             batch_dict_graph,
                                             n_traj_samples=3, kl_coef=kl_coef)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value, train_res["mse"], train_res["likelihood"], train_res["kl_first_p"], train_res["std_first_p"]


    def train_epoch(epo):
        model.train()
        loss_list = []
        mse_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            
            wait_until_kl_inc = 10

            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_feature_encoder = utils.get_next_batch_new(train_feature_encoder, device)
            batch_dict_edges_importance_encoder = utils.get_next_batch_new(train_edges_importance_encoder, device)
            batch_dict_observed_mask_encoder = utils.get_next_batch_new(train_observed_mask_encoder, device)
            batch_dict_avg_interval_encoder = utils.get_next_batch_new(train_avg_interval_encoder, device)
            batch_dict_time_pos_encoder = utils.get_next_batch_new(train_time_pos_encoder, device)
            batch_dict_length_encoder = utils.get_next_batch_new(train_length_encoder, device)

            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse, likelihood, kl_first_p, std_first_p = train_single_batch(model,
                                                                                batch_dict_feature_encoder,
                                                                                batch_dict_edges_importance_encoder,
                                                                                batch_dict_observed_mask_encoder,
                                                                                batch_dict_avg_interval_encoder,
                                                                                batch_dict_time_pos_encoder,
                                                                                batch_dict_length_encoder,
                                                                                batch_dict_decoder,
                                                                                batch_dict_graph,
                                                                                kl_coef)

        
            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(
                likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_feature_encoder, batch_dict_edges_importance_encoder, batch_dict_observed_mask_encoder, batch_dict_avg_interval_encoder, batch_dict_time_pos_encoder, batch_dict_length_encoder, batch_dict_graph, batch_dict_decoder
     
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            epo,
            np.mean(loss_list), np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list))

        return message_train, kl_coef


    for epo in range(1, args.niters + 1):

        message_train, kl_coef = train_epoch(epo)

        if epo % n_iters_to_viz == 0:
            model.eval()
            test_res = compute_loss_all_batches(model,
                                                test_feature_encoder,
                                                test_edges_importance_encoder,
                                                test_observed_mask_encoder,
                                                test_avg_interval_encoder,
                                                test_time_pos_encoder,
                                                test_length_encoder,
                                                test_graph,
                                                test_decoder,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)

            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                test_res["loss"], test_res["mse"], test_res["likelihood"],
                test_res["kl_first_p"], test_res["std_first_p"])

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            print("data: %s, sample: %s, mode:%s" % (
                args.data, str(args.sample_percent_train), args.mode))

            if test_res["mse"] < best_test_mse:
                best_test_mse = test_res["mse"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}|'.format(epo,
                                                                                                        best_test_mse)
                logger.info(message_best)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.data + "_" + str(
                    args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)

            torch.cuda.empty_cache()














