import argparse

parser = argparse.ArgumentParser(description='LSCI-DiffRec (Stage I+)')
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--score_lr', type=float, default=0.01,
                    help='Learning rate for the differentiable causal edge scorer.')
parser.add_argument('--wd2', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--epochs', '-e', type=int, default=25)
parser.add_argument('--hidden1', '-h1', type=int, default=8)
parser.add_argument('--hidden2', '-h2', type=int, default=8)
parser.add_argument('--dataset', '-d', type=str, default='yelp2018')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1024)
parser.add_argument('--emd_size', type=int, default=8)
parser.add_argument('--run_id', type=str, default='lsci_stage1')

parser.add_argument('--time_type', type=str, default='cat')
parser.add_argument('--mlp_dims', type=str, default='[8]')
parser.add_argument('--norm', type=bool, default=False)
parser.add_argument('--emb_size', type=int, default=10)
parser.add_argument('--mlp_act_func', type=str, default='tanh')

parser.add_argument('--mean_type', type=str, default='x0')
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--noise_schedule', type=str, default='linear-var')
parser.add_argument('--noise_scale', type=float, default=0.1)
parser.add_argument('--noise_min', type=float, default=0.01)
parser.add_argument('--noise_max', type=float, default=0.09)
parser.add_argument('--sampling_noise', type=bool, default=False)
parser.add_argument('--sampling_steps', type=int, default=100)
parser.add_argument('--reweight', type=bool, default=True)
parser.add_argument('--risk_weight_start', type=float, default=1.0)
parser.add_argument('--risk_weight_end', type=float, default=1.0)

# LSCI Stage I
parser.add_argument('--causal_keep_ratio', type=float, default=0.7)
parser.add_argument('--lambda_inv', type=float, default=0.1)
parser.add_argument('--inv_alpha', type=float, default=1.0)
parser.add_argument('--inv_beta', type=float, default=0.1)
parser.add_argument('--inv_mean_weight', type=float, default=0.0,
                    help='Optional risk-mean weight inside InvariantLoss; 0 avoids double-counting outer Mean.')
parser.add_argument('--score_warmup_epochs', type=int, default=3)
parser.add_argument('--variant_keep_prob', type=float, default=0.5,
                    help='Per-environment keep probability for the soft variant-edge gate.')
parser.add_argument('--edge_gate_mode', choices=['soft_pair_environment', 'random_pair_environment', 'none'],
                    default='none', help='Environment gate mode; none is the production baseline.')
parser.add_argument('--use_semantic_prior', action='store_true', default=False)
parser.add_argument('--semantic_prior_path', type=str, default='')
parser.add_argument('--lambda_sem', type=float, default=0.1, help='E3 InfoNCE weight')
parser.add_argument(
    '--semantic_score_alpha', type=float, default=0.0,
    help='Fixed validation-selected alpha for train-profile semantic late fusion.',
)
parser.add_argument(
    '--force_sem_gate', type=float, default=-1.0,
    help='If in [0,1], fix fusion gate g (smaller=more semantic). -1=learned gate.',
)
parser.add_argument('--generator_rank', type=int, default=128)
parser.add_argument('--num_env', type=int, default=3)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--fusion_lr', type=float, default=1e-3)
parser.add_argument(
    '--lambda_rank_upstream', type=float, default=0.0,
    help='Weight of differentiable BPR applied to upstream predicted embeddings.',
)
parser.add_argument(
    '--rank_batch_size', type=int, default=2048,
    help='Train triples sampled per environment for the upstream ranking objective.',
)
parser.add_argument(
    '--rank_batches_per_epoch', type=int, default=1,
    help='Number of upstream BPR batches per environment and epoch.',
)
parser.add_argument('--rank_layers', type=int, default=3)
parser.add_argument(
    '--rec_refresh', type=float, default=0.25,
    help='EMA refresh rate for the persistent ranking head from current upstream embeddings.',
)
parser.add_argument(
    '--rec_refresh_source', choices=['upstream', 'matched_random'], default='upstream',
    help='Refresh from Stage-I output or a distribution-matched random control.',
)
parser.add_argument('--rec_lr', type=float, default=0.001)
parser.add_argument('--rec_batch_size', type=int, default=1024)
parser.add_argument(
    '--rec_epochs_per_outer', type=int, default=1,
    help='Full BPR passes per outer epoch; keep identical to the E0 arm.',
)
parser.add_argument(
    '--allow_pre_split_selection', action='store_true', default=False,
    help='Allow selection before a causal/variant split exists (ablation only).',
)
parser.add_argument(
    '--validation_only', action='store_true', default=False,
    help='Select and report on validation only; never evaluate test GT.',
)

# Strict data (data_strict/processed/<ds>/v1_strict)
parser.add_argument(
    '--data_root', type=str, default='',
    help='Path to v1_strict root (must contain READY). Empty = legacy ./dataset/',
)
parser.add_argument(
    '--eval_split', type=str, default='ood', choices=['ood', 'iid'],
    help='Strict eval GT split: ood_test or iid_test parquet',
)

args = parser.parse_args()
