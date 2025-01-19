import torch
import numpy as np
from torchvision import datasets, transforms
import random

from flwr.common.secure_aggregation.quantization import quantize, dequantize

CLIPPING_RANGE =  3.0
TARGET_RANGE = 2**31

def get_dataset(args):
    """
    Returns the training and test datasets, along with a user group (dictionary).
    The dictionary maps each user index to their corresponding data.

    Args:
    - args: A set of arguments specifying the dataset type, IID/non-IID setting, and other parameters.

    Returns:
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - user_groups: A dictionary mapping user IDs to their respective dataset partitions.
    """

    # Load CIFAR-100 dataset
    if args.dataset == 'cifar100':
        data_dir = '../data/cifar100/'
        
        # Data transformations for training and testing
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Download CIFAR-100 training and test datasets
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)

    # Load EMNIST datasets
    elif args.dataset in ['mnist', 'emnist', 'emnist-balanced']:
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if args.dataset == 'emnist-balanced':
            data_dir = '../data/emnist-balanced/'
            # Download EMNIST (balanced split) datasets
            train_dataset = datasets.EMNIST(data_dir, split='balanced', train=True, download=True, transform=apply_transform)
            test_dataset = datasets.EMNIST(data_dir, split='balanced', train=False, download=True, transform=apply_transform)

    # Sample user data: IID or non-IID
    if args.iid:
        user_groups = distribute_dataset(train_dataset, args.num_users, args.num_samples_per_user)
    else:
        user_groups = distribute_dataset(train_dataset, args.num_users, args.num_samples_per_user, args.num_unique_classes_non_iid, args.num_non_iid_users, unequal=args.unequal)
        
    return train_dataset, test_dataset, user_groups

def distribute_dataset(dataset, num_users: int, num_samples_per_user: int | None, num_unique_classes_non_iid=0, num_non_iid_users=0, unequal=False):
    """Distributes the dataset to users in a non-IID manner.
    
    Args:
    - dataset: The dataset.
    - num_users: The number of users to distribute data to.
    - num_unique_classes_non_iid: The number of classes per user in the non-iid group.
    - num_non_iid_users: The number of users in the non-iid group.
    
    Returns:
    - A dictionary mapping user IDs to their assigned indices.
    """
    
    dict_users = {i + 1: np.array([]) for i in range(num_users)} # Initialize user dictionary
    labels = np.array(dataset.targets) # Extract labels from dataset
    num_classes = len(set(labels))
    
    assert num_unique_classes_non_iid * num_non_iid_users < num_classes, \
        f'not enough labels {num_unique_classes_non_iid * num_non_iid_users} for {num_classes}'
    
    non_iid_classes = [] # Track classes already assigned
    for non_iid_user_id in range(1, num_non_iid_users + 1):
        unique_classes = []
        for _ in range(num_unique_classes_non_iid):
            rand_class = random.randint(0, num_classes - 1)
            while rand_class in non_iid_classes: # Ensure unique class selection
                rand_class = random.randint(0, num_classes - 1)
            non_iid_classes.append(rand_class) # Add to tracked classes
            unique_classes.append(rand_class)
        samples_in_unique_classes = np.array([idx for idx in range(len(labels)) if labels[idx] in unique_classes])
        if num_samples_per_user is None:
            dict_users[non_iid_user_id] = samples_in_unique_classes
        else:
            if len(samples_in_unique_classes) <= num_samples_per_user or unequal:
                dict_users[non_iid_user_id] = samples_in_unique_classes
            else:
                dict_users[non_iid_user_id] = np.random.choice(samples_in_unique_classes, num_samples_per_user, replace=False)

    remaining_idxs = [idx for idx in range(len(labels)) if labels[idx] not in non_iid_classes]
    if num_samples_per_user is None:
        num_samples_per_user = int(round(len(remaining_idxs) / (num_users - num_non_iid_users)))
    else:
        if len(remaining_idxs) < (num_users - num_non_iid_users) * num_samples_per_user:
            num_samples_per_user = int(round(len(remaining_idxs) / (num_users - num_non_iid_users)))

    # Assign remaining users with random indices
    for user_id in range(num_non_iid_users + 1, num_users + 1):
        selected_idxs = np.random.choice(remaining_idxs, num_samples_per_user, replace=False)
        dict_users[user_id] = selected_idxs # Assign indices to user
        remaining_idxs = np.setdiff1d(remaining_idxs, selected_idxs) # Update available indices

    return dict_users

def count_trainable_params(model_dict):
    trainable_params = 0
    for params in model_dict.keys():
        trainable_params += model_dict[params].numel()
    return trainable_params

def quantize_vector(vector, clipping_range, target_range):
    return quantize([vector], clipping_range, target_range)[0]

def dequantize_vector(vector, clipping_range, target_range):
    return dequantize([np.array(vector)], clipping_range, target_range)[0]

def reshape_back_to_dict(flat_list, template_dict):
    new_dict = {}
    idx = 0
    for key in sorted(template_dict.keys()):
        num_elements = template_dict[key].numel()
        new_tensor = torch.tensor(flat_list[idx:idx + num_elements]).reshape(template_dict[key].shape)
        new_dict[key] = new_tensor
        idx += num_elements
    return new_dict

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model: {args.model}')
    print(f'    Optimizer: {args.optimizer}')
    print(f'    Learning rate: {args.lr}')
    print(f'    Global rounds: {args.epochs}')
    print(f'    Local batch size: {args.local_bs}')
    print(f'    Local epochs: {args.local_ep}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        print(f'    Number of unique classes in non-iid group: {args.num_unique_classes_non_iid}')
        print(f'    Number of non-iid users: {args.num_non_iid_users}')
    print(f'    Aggregation algorithm: {args.agg_type} {args.fedavg_type if args.agg_type == 'fedavg' else ''}')
    print(f'    Secure aggregation: {args.sec_agg}')
    print(f'    Participation rate: {args.frac}')
    print(f'    Dropout users: {args.dropout}')
    if args.dropout > 0:
        print(f'    First epoch with dropouts: {args.first_dropout_epoch}')
        print(f'    Last epoch with dropouts: {args.last_dropout_epoch}')
    if args.dropout > 0 or args.frac < 1:
        print(f'    Use approximated model updates: {args.use_approx_updates}')
    print(f'    Local DP: {args.local_dp}')
    if args.local_dp:
        print(f'        iid_epsilon: {args.iid_epsilon}, non_iid_epsilon: {args.non_iid_epsilon}')
        print(f'        delta: {args.delta}')
        print(f'        max_grad_norm: {args.max_grad_norm}')