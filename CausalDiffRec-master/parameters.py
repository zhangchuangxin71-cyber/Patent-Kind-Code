import argparse
parser = argparse.ArgumentParser(description='Variant Graph Autoencoder')
parser.add_argument('--lr2', type=float, default=0.1, help='learning rate for MLP')
parser.add_argument('--wd2', type=float, default=0.0, help='weight decay for MLP')

parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=25, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=8, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=8, help='Number of units in hidden layer 2.')
parser.add_argument('--dataset', '-d', type=str, default='yelp2018', help='Dataset string.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility.')
parser.add_argument('--emd_size', type=int, default=8, help='the dims size')
parser.add_argument('--run_id', type=str, default='', help='Optional run id tag for experiment logs.')
parser.add_argument(
    '--data_root', type=str, default='',
    help='Path to v1_strict root (must contain READY). Empty = legacy ./dataset/',
)
parser.add_argument(
    '--eval_split', type=str, default='ood', choices=['ood', 'iid'],
    help='Strict eval split when --data_root is set.',
)
parser.add_argument(
    '--validation_only', action='store_true',
    help='Select/checkpoint on validation only; do not load or evaluate test GT.',
)

# params for the MLP
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--mlp_dims', type=str, default='[8]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')
parser.add_argument('--optimizer2', type=str, default='AdamW',
                    help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.01)
parser.add_argument('--noise_max', type=float, default=0.09)
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=100, help='steps for sampling/denoising')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--grad_clip', type=float, default=5.0, help='Global gradient-norm clipping threshold.')
parser.add_argument('--risk_weight_start', type=float, default=1.0)
parser.add_argument('--risk_weight_end', type=float, default=1.0)
parser.add_argument('--rec_refresh', type=float, default=0.25)
parser.add_argument('--rec_lr', type=float, default=0.001)
parser.add_argument('--rec_batch_size', type=int, default=1024)
parser.add_argument(
    '--rec_epochs_per_outer', type=int, default=1,
    help='Full BPR passes per outer epoch; use 8 with 25 outer epochs for 200 ranking passes.',
)

args = parser.parse_args()
