import os
import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import csv
import random
from tqdm import tqdm
from datetime import datetime

from options import args_parser # Argument parser for command-line options
from models import (
    CNNEmnist, # CNN model for the EMNIST dataset
    CNNCifar100, # CNN model for the CIFAR-100 dataset
)
from utils import (
    get_dataset, # Function to load datasets
    exp_details, # Function to print experiment details
)

from fl_settings import (
    Client,
    Server,
    DatasetSplit,
)

if __name__ == '__main__':
    start_time = time.time() # Start timer for performance measurement

    # Define paths
    path_project = os.path.abspath('..') # Get the absolute path of the project
    logger = None # Placeholder for TensorBoard logger (currently unused)

    # Parse command-line arguments
    args = args_parser()
    assert args.num_non_iid_users < args.num_users, 'Bad inputs'
    if args.dropout > 0:
        assert args.first_dropout_epoch <= args.last_dropout_epoch, 'Bad inputs'
        assert args.first_dropout_epoch > 0 and args.last_dropout_epoch > 0, 'Bad inputs'
        assert args.first_dropout_epoch <= args.epochs and args.last_dropout_epoch <= args.epochs, 'Bad inputs'

    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Load datasets and user groups
    user_ids = list(range(1, args.num_users + 1))
    train_dataset, test_dataset, user_groups = get_dataset(args)
    num_classes = len(set(np.array(train_dataset.targets)))
    num_test_samples = len(test_dataset.targets)
    
    now = datetime.now()
    current_time = now.strftime('%Y_%m_%d_%H_%M_%S') # Get current timestamp for logging
    start_time = time.time()
    
    num_train_samples = 0
    num_samples_per_user = 0

    if not os.path.isdir('values'):
        os.mkdir('values')
    with open(f'values/user_groups_{args.agg_type}_num_unique_classes_non_iid{args.num_unique_classes_non_iid}_num_non_iid_users{args.num_non_iid_users}_{current_time}.txt', 'w') as f:
        for user_id in user_ids:
            print(f'user_id: {user_id}', file=f)
            samples_ids = [int(s) for s in user_groups[user_id]]
            samples_ids.sort()

            samples_classes = [int(train_dataset.targets[i]) for i in samples_ids]
            samples_classes.sort()
            num_train_samples += len(samples_classes)
            if user_id == args.num_non_iid_users + 1:
                num_samples_per_user = len(samples_classes)
            print(f'Samples Size: {len(samples_classes)}', file=f)
            print(f'Samples Classes: {set(samples_classes)}', file=f)

    exp_details(args) # Print experiment details
    exp_results = vars(args)
    print(f'Device: {device}')
    if args.iid:
        iidness = 'iid'
    else:
        iidness = 'noniid'
    my_path = os.getcwd()
    
    # BUILD MODEL
    if args.model == 'cnn':
        # Choose model based on dataset
        if args.dataset == 'emnist-balanced':
            global_model = CNNEmnist(args, num_classes)
        elif args.dataset == 'cifar100':
            global_model = CNNCifar100(args, num_classes)
    else:
        exit('Error: Unrecognized model')

    # Move model to the appropriate device and set to training mode
    global_model.to(device)
    global_model.train()

    # Initialize lists to track losses and accuracies
    train_loss, train_accuracy = [], []
    test_loss_list, test_accuracy_list = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    sec_agg_model_updates_time_server, sec_agg_model_updates_time_client = [], []
    sec_agg_model_updates_data_transfer, sec_agg_model_updates_data_transfer_client = [], []
    print_every = 1 # Frequency of logging results
    val_loss_pre, counter = 0, 0 # Variables for early stopping logic
    
    clients = {
        user_id: Client(user_id, args, DatasetSplit(train_dataset, user_groups[user_id]), num_classes, copy.deepcopy(global_model), logger)
            for user_id in user_ids
    }
    server = Server(args, test_dataset, num_classes, copy.deepcopy(global_model), clients, user_ids)
    
    if args.agg_type == 'fedla':
        if args.sec_agg:
            sec_agg_setup_time_server, sec_agg_setup_time_client = server.setup_sec_agg(clients, user_ids)
            exp_results['sec_agg_setup_time_server'] = sec_agg_setup_time_server
            exp_results['sec_agg_setup_time_client'] = sec_agg_setup_time_client
            total_samples_per_class, sec_agg_label_time_server, sec_agg_label_time_client, sec_agg_data_transfer, sec_agg_data_transfer_client = server.aggregate_total_samples_per_class(clients, user_ids)
            exp_results['sec_agg_label_time_server'] = sec_agg_label_time_server
            exp_results['sec_agg_label_time_client'] = sec_agg_label_time_client
            exp_results['sec_agg_label_data_transfer'] = sec_agg_data_transfer
            exp_results['sec_agg_label_data_transfer_client'] = sec_agg_data_transfer_client
            for user_id in user_ids:
                clients[user_id].calculate_weight(total_samples_per_class)

    if not os.path.isdir('../results'):
        os.mkdir('../results')

    for epoch in range(args.epochs):
        print(f'\nGlobal training round: {epoch + 1}')
        print(f'Aggregation algorithm: {args.agg_type} {args.fedavg_type if args.agg_type == 'fedavg' else ''}')
        print(f'Using SecAgg: {args.sec_agg}')
        print(f'Using LocalDP (Opacus): {args.local_dp}')
        if args.local_dp:
            print(f'    C={args.max_grad_norm}, iid_epsilon={args.iid_epsilon}, non_iid_epsilon={args.non_iid_epsilon}, delta={args.delta}')
        
        local_losses = [] # Initialize lists for local weights and losses
        global_model.train() # Set model to training mode
        
        num_participating_users = int(round(args.num_users * args.frac))
        participating_ids = np.random.choice(user_ids, num_participating_users, replace=False).tolist()
        if epoch + 1 >= args.first_dropout_epoch and epoch + 1 <= args.last_dropout_epoch and args.dropout > 0:
            alive_ids = np.random.choice(participating_ids, num_participating_users - args.dropout, replace=False).tolist()
        else:
            alive_ids = participating_ids
        participating_ids.sort()
        alive_ids.sort()
        num_alive_users = len(alive_ids)
        print(f'Participating users: {participating_ids}')
        print(f'Alive users: {alive_ids}')
        if num_alive_users < args.num_users and args.sec_agg and args.agg_type == 'fedla':
            print(f'Use approximated model updates: {args.use_approx_updates}')
        
        local_model_time = 0
        for user_id in tqdm(alive_ids, desc=f'Local training ({args.local_ep} epochs, lr={args.lr})'):
            start = time.time()
            loss = clients[user_id].train(global_model.state_dict())
            local_losses.append(copy.deepcopy(loss)) # Store local loss
            end = time.time()
            local_model_time += end - start
        print(f'Avg local model time: {local_model_time / num_alive_users:.2f}s')

        # Update global weights using either simple FedAvg or FedLA
        if args.agg_type == 'fedla' and args.sec_agg:
            aggregated_model_updates, sec_agg_time_server, sec_agg_time_client, sec_agg_data_transfer, sec_agg_data_transfer_client = server.aggregate_model_updates(clients, participating_ids, alive_ids, global_model.state_dict())
            sec_agg_model_updates_time_server.append(sec_agg_time_server)
            sec_agg_model_updates_time_client.append(sec_agg_time_client)
            sec_agg_model_updates_data_transfer.append(sec_agg_data_transfer)
            sec_agg_model_updates_data_transfer_client.append(sec_agg_data_transfer_client)
        else:
            aggregated_model_updates = server.aggregate_model_updates(clients, participating_ids, alive_ids, global_model.state_dict())
        
        # Load the averaged weights into the global model
        global_model.load_state_dict(aggregated_model_updates)

        loss_avg = sum(local_losses) / len(local_losses) # Average loss from local updates
        train_loss.append(round(loss_avg, 3)) # Store average training loss

        # Calculate average training accuracy across all users at every epoch
        list_acc, list_loss = [], [] # Lists for tracking accuracy and loss
        for user_id in user_ids:
            acc, loss = clients[user_id].inference(global_model.state_dict())
            list_acc.append(acc) # Store accuracy
            list_loss.append(loss) # Store loss
            
        train_accuracy.append(
            round((sum([clients[user_id].get_num_samples() * list_acc[user_id - 1] for user_id in user_ids]) / num_train_samples), 4) # Average training accuracy
        )

        # Test inference after completing training for this round
        test_acc, test_loss = server.test_inference(global_model.state_dict()) # Evaluate on the test set
        test_accuracy_list.append(test_acc) # Store test accuracy
        test_loss_list.append(test_loss / 100) # Store test loss (scaled)

        # Print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f'\nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Train Accuracy: {100 * train_accuracy[-1]:.2f}%')
            print(f'Training Loss: {np.mean(np.array(train_loss))}')
            print(f'Test Accuracy: {100 * test_acc:.2f}%')
            print(f'Test Loss: {round((test_loss / 100), 3)}')

            with open(os.path.join(my_path, f'../results/{args.dataset}_{iidness}_epochs{args.epochs}_local_ep{args.local_ep}_lr{args.lr}_{args.agg_type}{f'_{args.fedavg_type}' if args.agg_type == 'fedavg' else ''}{f'_sci{num_samples_per_user}'}{f'_frac{args.frac}'}{f'_uniqueC{args.num_unique_classes_non_iid}_noniidS{args.num_non_iid_users}' if not args.iid else ''}{'_unequal' if args.unequal else ''}{'_secagg' if args.sec_agg else ''}{f'_dropout{args.dropout}' if args.dropout > 0 else ''}{f'_use_approx_updates' if args.use_approx_updates and (args.dropout > 0 or args.frac < 1) else ''}{f'_local_dp_C{args.max_grad_norm}_iid_epsilon{args.iid_epsilon}_non_iid_epsilon{args.non_iid_epsilon}_delta{args.delta}' if args.local_dp else ''}.csv'), 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(test_accuracy_list)

    #######################   PLOTTING & args & results saving    ###################################

    plt.switch_backend('Agg')
    
    full_path = os.path.join(
        my_path,
        f'../results/{args.dataset}/{iidness}/{args.agg_type}/{args.epochs}/{args.num_unique_classes_non_iid}/{args.num_non_iid_users}/{current_time}',
    )
    os.makedirs(full_path, exist_ok=True)

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/training_loss.pdf')

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/training_accuracy.pdf')

    plt.figure()
    plt.title('Testing Loss vs Communication rounds')
    plt.plot(range(len(test_loss_list)), test_loss_list, color='r')
    plt.ylabel('Testing loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/testing_loss.pdf')

    plt.figure()
    plt.title('Testing Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, color='k')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/testing_accuracy.pdf')

    # YAML file with all data and results
    if args.iid:
        exp_results.pop('num_unique_classes_non_iid', None)
        exp_results.pop('num_non_iid_users', None)
    if args.agg_type != 'fedavg':
        exp_results.pop('fedavg_type', None)
    if not args.dropout > 0:
        exp_results.pop('first_dropout_epoch', None)
        exp_results.pop('last_dropout_epoch', None)
    if not args.local_dp:
        exp_results.pop('secure_rng', None)
        exp_results.pop('max_grad_norm', None)
        exp_results.pop('iid_epsilon', None)
        exp_results.pop('non_iid_epsilon', None)
        exp_results.pop('delta', None)
        
    exp_results['train_accuracy'] = train_accuracy
    exp_results['train_loss'] = train_loss
    exp_results['avg_train_accuracy'] = round(train_accuracy[-1], 3)
    exp_results['avg_train_loss'] = round((train_loss[-1] / 100), 3)
    exp_results['test_accuracy_list'] = test_accuracy_list
    exp_results['test_loss_list'] = test_loss_list
    
    if args.agg_type == 'fedla' and args.sec_agg:
        exp_results['sec_agg_model_updates_time_server'] = sec_agg_model_updates_time_server
        exp_results['sec_agg_model_updates_time_client'] = sec_agg_model_updates_time_client
        exp_results['sec_agg_model_updates_data_transfer'] = sec_agg_model_updates_data_transfer
        exp_results['sec_agg_model_updates_data_transfer_client'] = sec_agg_model_updates_data_transfer_client
    
    with open(f'{full_path}/data.yml', 'w') as outfile:
        yaml.dump(exp_results, outfile, default_flow_style=False)
