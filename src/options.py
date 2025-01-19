import argparse

def args_parser():
    """Parses command-line arguments for the federated learning setup."""
    
    parser = argparse.ArgumentParser()

    # Federated learning arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of rounds of training (default: 50).')
    parser.add_argument('--num_users', type=int, default=10,
                        help='Total number of users participating in federated learning (default: 10).')
    parser.add_argument('--local_ep', type=int, default=1,
                        help='Number of local epochs to train on each user (default: 1).')
    parser.add_argument('--local_bs', type=int, default=256,
                        help='Local batch size for training (default: 256).')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for optimization (default: 0.1).')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='Momentum for SGD optimization (default: 0.5).')

    # Model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        help='Name of the model to use (default: cnn).')

    # Other arguments
    parser.add_argument('--dataset', type=str, default='emnist-balanced',
                        help='Name of the dataset (e.g., emnist-balanced, cifar100) (default: emnist-balanced).')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Type of optimizer to use (e.g., sgd, adam) (default: sgd).')
    parser.add_argument('--iid', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable IID data distribution.')
    parser.add_argument('--unequal', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable unequal data splits between IID and non-IID group.')
    parser.add_argument('--num_samples_per_user', type=int, default=None,
                        help='Number of samples distributed to each user in the IID group.')
    parser.add_argument('--num_unique_classes_non_iid', type=int, default=0,
                        help='Unique count of classes of each user in the non-IID group.')
    parser.add_argument('--num_non_iid_users', type=int, default=0,
                        help='Number of non-IID users (default: 0).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility.')
    parser.add_argument('--agg_type', type=str, default='fedavg',
                        help='Method for federated aggregation: fedavg or fedla (default: fedavg).')
    parser.add_argument('--fedavg_type', type=str, default='unweighted',
                        help='Type of fedavg: weighted or unweighted (default: unweighted).')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='The participation ratio at each round, k_r (default: 1.0).')
    
    # Dropout and dropout handling arguments
    parser.add_argument('--dropout', type=int, default=0,
                        help='The number of users dropping out at each round (default: 0).')
    parser.add_argument('--first_dropout_epoch', type=int, default=1,
                        help='The epoch when client dropouts first occur.')
    parser.add_argument('--last_dropout_epoch', type=int, default=50,
                        help='The epoch when client dropouts last occur.')
    parser.add_argument('--use_approx_updates', action=argparse.BooleanOptionalAction, default=False,
                        help='Use approximated model updates for non-participating and/or dropout clients instead of the encryptions of 0-value when using secure aggregation')
    parser.add_argument('--sec_agg', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable secure aggregation.')
    
    # Differential privacy arguments
    parser.add_argument('--local_dp', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable local differential privacy.')
    parser.add_argument('--secure_rng', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable secure random number generator for local differential privacy with opacus')
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Gradient clipping threshold for local differential privacy, (default: 3.0).')
    parser.add_argument('--iid_epsilon', type=float, default=40.0,
                        help='Target epsilon for local differential privacy of iid users (default: 40.0).')
    parser.add_argument('--non_iid_epsilon', type=float, default=10.0,
                        help='Target epsilon for local differential privacy of non-iid users (default: 10.0).')
    parser.add_argument('--delta', type=float, default=1e-6,
                        help='target delta for local differential privacy, (default: 1e-6).')

    args = parser.parse_args()
    print(args)
    return args
