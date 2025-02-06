import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine

from ftsa.protocols.ourftsa22.client import Client as FTSA_Client
from ftsa.protocols.ourftsa22.server import Server as FTSA_Server
from ftsa.protocols.buildingblocks.JoyeLibert import PublicParam, TJLS

from sys import getsizeof
from utils import (
    count_trainable_params,
    quantize_vector,
    dequantize_vector,
    reshape_back_to_dict,
    CLIPPING_RANGE,
    TARGET_RANGE
)

from typing import Any, Mapping, List
import time
import copy
from tqdm import tqdm
from math import ceil

class DatasetSplit(Dataset):
    """An abstract Dataset class that wraps around the PyTorch Dataset class.
    
    This class is used to create a subset of a dataset based on given indices.
    """

    def __init__(self, dataset, idxs):
        """
        Initializes the DatasetSplit instance.
        
        Args:
        - dataset: The original dataset to split.
        - idxs: A list of indices corresponding to the data points in the dataset.
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs] # Convert indices to integers

    def __len__(self):
        """Returns the number of items in the dataset subset."""
        return len(self.idxs)

    def __getitem__(self, item):
        """Fetches the item at the specified index from the dataset.
        
        Args:
        - item: Index of the item to fetch.
        
        Returns:
        - A tuple (image, label) as tensors.
        """
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label) # Return tensors

@staticmethod
def setup_sec_agg(dimension: int, input_size: int, key_size: int, threshold: int, num_users, public_params: PublicParam):
    FTSA_Client.set_scenario(dimension, input_size, key_size, threshold, num_users, public_params)
    FTSA_Server.set_scenario(dimension, input_size, key_size, threshold, num_users, public_params)

class Client(object):
    def __init__(self, user_id: int, args, user_train_dataset: DatasetSplit, num_classes: int, model: nn.Module, logger):
        self.user_id = user_id
        self.args = args
        self.logger = logger
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device to GPU or CPU
        self.__model = model
        self.optimizer = torch.optim.SGD(self.__model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.train_dataset = user_train_dataset
        self.__train_loader = DataLoader(self.train_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.criterion = nn.CrossEntropyLoss() # Define loss function for training
        
        self.__num_samples = len(user_train_dataset)
        self.__samples_per_class = [0] * num_classes
        for i in range(len(user_train_dataset.idxs)):
            _, label = user_train_dataset[i]
            self.__samples_per_class[int(label)] += 1
        self.__num_classes = num_classes - self.__samples_per_class.count(0)
        self.iid = False if self.__num_classes == self.args.num_unique_classes_non_iid else True
    
    def init_sec_agg(self, ftsa_uid: int):
        self.ftsa_uid = ftsa_uid
        self.ftsa_client = FTSA_Client(self.ftsa_uid)
        
    def get_samples_per_class(self, step=0):
        if self.args.sec_agg:
            self.ftsa_client.new_fl_step(self.__samples_per_class, step)
            _, eb_shares, encrypted_samples_per_class = self.ftsa_client.online_encrypt()
            return eb_shares, encrypted_samples_per_class
        else:
            return self.__samples_per_class
    
    def get_num_samples(self):
        return self.__num_samples
    
    def get_model_update(self, step=None):
        model_update = self.__model.state_dict()
        
        if self.args.sec_agg:
            flattened_model_update = torch.Tensor([]).to(self.device)
            for params in sorted(model_update.keys()):
                flattened_model_update = torch.cat((flattened_model_update, torch.flatten(model_update[params] * self.__weight)))
            quantized_model_update = quantize_vector(flattened_model_update.tolist(), CLIPPING_RANGE, TARGET_RANGE).tolist()
            
            self.ftsa_client.new_fl_step(quantized_model_update, step)
            _, eb_shares, encrypted_quantized_model_update = self.ftsa_client.online_encrypt()
            return eb_shares, encrypted_quantized_model_update
        else:
            return model_update
        
    def get_dropout_approx_share(self, all_user_eb_shares, num_alive_users: int, global_model: Mapping[str, Any]):
        assert self.args.agg_type == 'fedla' and self.args.sec_agg == True, 'This method is only available in SecAgg mode.'
        flattened_global_model = torch.Tensor([]).to(self.device)
        for params in sorted(global_model.keys()):
            flattened_global_model = torch.cat((flattened_global_model, torch.flatten(global_model[params])))

        dropout_approx_share = ((1 / num_alive_users) - self.__weight) * flattened_global_model
        quantized_dropout_approx_share = quantize_vector(dropout_approx_share.tolist(), CLIPPING_RANGE, TARGET_RANGE).tolist()
        _, b_shares, encrypted_quantized_dropout_approx_share = self.ftsa_client.online_construct(all_user_eb_shares, quantized_dropout_approx_share)
        
        return b_shares, encrypted_quantized_dropout_approx_share
    
    def get_weight(self):
        return self.__weight
    
    def calculate_weight(self, total_samples_per_class: List[int]):
        assert self.args.agg_type == 'fedla' and self.args.sec_agg == True, 'This method is only available in SecAgg mode.'
        user_power = {}
        for label in range(len(self.__samples_per_class)):
            # W(c_i, l_j) = S(c_i, l_j) / S(l_j)
            if total_samples_per_class[label] != 0:
                user_power[label] = self.__samples_per_class[label] / total_samples_per_class[label]
            else:
                user_power[label] = 0
        
        # W(c_i) = SUM(W(c_i, l_j))
        user_power = sum(user_power.values())
        total_power = len(total_samples_per_class) - total_samples_per_class.count(0)
        # W_FedLA(c_i) = W(c_i) / SUM(W(c_i))
        self.__weight = user_power / total_power
    
    def train(self, global_model: Mapping[str, Any]):
        """Trains the model using the local training data and updates the weights.
        
        Args:
        - global_model: The weights of the model to be trained.
        
        Returns:
        - avg_epoch_loss: The average epoch loss.
        """
        self.__model.load_state_dict(global_model)
        self.__model.train()
        
        if self.args.local_dp:
            self.optimizer = torch.optim.SGD(self.__model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            self.__train_loader = DataLoader(self.train_dataset, batch_size=self.args.local_bs, shuffle=True)
            privacy_engine = PrivacyEngine(secure_mode=self.args.secure_rng)
            
            epsilon = self.args.iid_epsilon if self.iid else self.args.non_iid_epsilon
            
            self.__model, self.optimizer, self.__train_loader = privacy_engine.make_private_with_epsilon(
                module = self.__model,
                optimizer = self.optimizer,
                data_loader = self.__train_loader,
                target_epsilon=epsilon,
                target_delta=self.args.delta,
                epochs=self.args.local_ep,
                max_grad_norm = self.args.max_grad_norm
            )
        epoch_loss = []
        
        for _ in range(self.args.local_ep):
            batch_loss = [] # List to store loss values for the current batch
            for _, (images, labels) in enumerate(self.__train_loader):
                images, labels = images.to(self.device), labels.to(self.device) # Move data to device

                self.optimizer.zero_grad() # Zero the gradients before backward pass
                log_probs = self.__model(images) # Forward pass
                loss = self.criterion(log_probs, labels) # Compute loss
                loss.backward() # Backward pass
                self.optimizer.step() # Update weights
                batch_loss.append(loss.item()) # Record batch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss)) # Average loss for the epoch
        
        if self.args.local_dp:
            self.__model = self.__model.to_standard_module()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        # Return average loss
        return avg_epoch_loss

    def inference(self, global_model: Mapping[str, Any]):
        """Evaluates the global model on the local dataset and calculates accuracy and loss.
        
        Args:
        - global_model: The weights of global model to evaluate.
        
        Returns:
        - A tuple containing accuracy and total loss on the test dataset.
        """
        self.__model.load_state_dict(global_model)
        self.__model.eval() # Set model to evaluation mode
        loss, total, correct = 0.0, 0.0, 0.0 # Initialize metrics

        # Evaluate the model on the test dataset
        for _, (images, labels) in enumerate(self.__train_loader):
            images, labels = images.to(self.device), labels.to(self.device) # Move data to device

            outputs = self.__model(images) # Forward pass
            batch_loss = self.criterion(outputs, labels) # Compute loss
            loss += batch_loss.item() # Accumulate loss

            # Get predictions and calculate accuracy
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1) # Flatten predictions
            correct += torch.sum(torch.eq(pred_labels, labels)).item() # Count correct predictions
            total += len(labels) # Total number of labels processed

        accuracy = correct / total # Calculate accuracy
        return accuracy, loss # Return accuracy and loss

class Server(object):
    def __init__(self, args, test_dataset: Dataset, num_classes: int, model: nn.Module, clients: Mapping[int, Client], user_ids: List[int]):
        self.args = args
        self.__test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False) # Create test DataLoader
        self.num_classes = num_classes
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device for evaluation
        self.criterion = nn.CrossEntropyLoss()
        self.step = 0
        self.total_samples_per_user = {user_id: clients[user_id].get_num_samples() for user_id in user_ids}
        if not self.args.sec_agg:
            self.total_samples_per_class_per_user = {user_id: clients[user_id].get_samples_per_class() for user_id in user_ids}
    
    # def init_sec_agg(self, ):
    #     self.ftsa_server = FTSA_Server()
    
    def init_sec_agg(self, server_key):
        self.ftsa_server = FTSA_Server(server_key)
    
    def setup_sec_agg(self, clients: Mapping[int, Client], participating_ids: List[int]):
        assert self.args.sec_agg == True, 'This method is only available in SecAgg.'
        print('Setting up FTSA scheme for SecAgg')
        key_size, input_size, dimension, threshold = 2048, 32, count_trainable_params(self.model.state_dict()), ceil(0.6 * len(participating_ids))
        # public_params, _, _ = TJLS(len(participating_ids), threshold).Setup(key_size)
        public_params, server_key, user_keys = TJLS(len(participating_ids), threshold).Setup(key_size)
        setup_sec_agg(dimension, input_size, key_size, threshold, len(participating_ids), public_params)
        
        sec_agg_setup_time_server = time.time()
        # self.init_sec_agg()
        self.init_sec_agg(server_key)
        # all_pk_s, all_pk_c = {}, {}
        all_pk_c = {}
        sec_agg_setup_time_server = time.time() - sec_agg_setup_time_server
        
        sec_agg_setup_time_client = time.time()
        for ftsa_uid, user_id in enumerate(participating_ids, start=1):
            clients[user_id].init_sec_agg(ftsa_uid)
            # _, pk_s, pk_c = clients[user_id].ftsa_client.setup_register()
            _, pk_c = clients[user_id].ftsa_client.setup_register(user_keys[user_id - 1])
            # all_pk_s[clients[user_id].ftsa_uid] = pk_s
            all_pk_c[clients[user_id].ftsa_uid] = pk_c
        sec_agg_setup_time_client = time.time() - sec_agg_setup_time_client
        # sec_agg_setup_data_transfer = sum([getsizeof(all_pk_s[clients[user_id].ftsa_uid]) + getsizeof(all_pk_c[clients[user_id].ftsa_uid]) for user_id in participating_ids])
        sec_agg_setup_data_transfer = sum([getsizeof(all_pk_c[clients[user_id].ftsa_uid]) for user_id in participating_ids])
        
        sec_agg_setup_time_server -= time.time()
        # all_pk_c, all_pk_s = self.ftsa_server.setup_register(all_pk_c, all_pk_s)
        all_pk_c = self.ftsa_server.setup_register(all_pk_c)
        all_sk_shares = {}
        sec_agg_setup_time_server += time.time()
        # sec_agg_setup_data_transfer += sum([getsizeof(all_pk_s[clients[user_id].ftsa_uid]) + getsizeof(all_pk_c[clients[user_id].ftsa_uid]) for user_id in participating_ids]) * len(participating_ids)
        sec_agg_setup_data_transfer += sum([getsizeof(all_pk_c[clients[user_id].ftsa_uid]) for user_id in participating_ids]) * len(participating_ids)
        
        sec_agg_setup_time_client -= time.time()
        for user_id in participating_ids:
            # _, sk_shares = clients[user_id].ftsa_client.setup_keysetup(all_pk_s, all_pk_c)
            _, sk_shares = clients[user_id].ftsa_client.setup_keysetup(all_pk_c)
            all_sk_shares[clients[user_id].ftsa_uid] = sk_shares
        sec_agg_setup_time_client += time.time()
        sec_agg_setup_data_transfer += sum([getsizeof(all_sk_shares[clients[user_id].ftsa_uid]) for user_id in participating_ids])
        
        sec_agg_setup_time_server -= time.time()
        all_sk_shares = self.ftsa_server.setup_keysetup(all_sk_shares)
        sec_agg_setup_time_server += time.time()
        sec_agg_setup_data_transfer += sum([getsizeof(all_sk_shares[clients[user_id].ftsa_uid]) for user_id in participating_ids])
        
        sec_agg_setup_time_client -= time.time()
        for user_id in participating_ids:
            clients[user_id].ftsa_client.setup_keysetup2(all_sk_shares[clients[user_id].ftsa_uid])
        sec_agg_setup_time_client += time.time()
        print(f'SecAgg setup time (server): {sec_agg_setup_time_server * 1000000:.0f} μs')
        print(f'SecAgg setup time (avg per client): {sec_agg_setup_time_client * 1000 / len(participating_ids):.0f} ms')
        print(f'SecAgg setup data transfer (per client): {sec_agg_setup_data_transfer / len(participating_ids):.0f} bytes')
        return sec_agg_setup_time_server, (sec_agg_setup_time_client / len(participating_ids))
    
    def aggregate_total_samples_per_class(self, clients: Mapping[int, Client], participating_ids: List[int]):
        assert self.args.sec_agg == True, 'This method is only available in SecAgg.'
        self.step += 1
        self.ftsa_server.new_fl_step(self.step)
        
        sec_agg_client_time = time.time()
        all_eb_shares = {}
        encrypted_samples_per_class_per_user = {}
        for user_id in participating_ids:
            eb_shares, encrypted_user_samples_per_class = clients[user_id].get_samples_per_class(self.step)
            all_eb_shares[clients[user_id].ftsa_uid] = eb_shares
            encrypted_samples_per_class_per_user[clients[user_id].ftsa_uid] = encrypted_user_samples_per_class
        sec_agg_client_time = time.time() - sec_agg_client_time
        enc_round_client_msg_sizes = [getsizeof(all_eb_shares[clients[user_id].ftsa_uid]) + getsizeof(encrypted_samples_per_class_per_user[clients[user_id].ftsa_uid]) for user_id in participating_ids]
        
        sec_agg_server_time = time.time()
        all_eb_shares = self.ftsa_server.online_encrypt(all_eb_shares, encrypted_samples_per_class_per_user)
        sec_agg_server_time = time.time() - sec_agg_server_time
        enc_round_server_msg_sizes = [getsizeof(all_eb_shares[clients[user_id].ftsa_uid]) for user_id in participating_ids]
    
        sec_agg_client_time -= time.time()
        all_b_shares = {}
        encrypted_zero_shares = {}
        for user_id in participating_ids:
            _, b_shares, encrypted_zero_share = clients[user_id].ftsa_client.online_construct(all_eb_shares[clients[user_id].ftsa_uid], [0] * self.num_classes)
            all_b_shares[clients[user_id].ftsa_uid] = b_shares
            encrypted_zero_shares[clients[user_id].ftsa_uid] = encrypted_zero_share
        sec_agg_client_time += time.time()
        agg_round_client_msg_size = [getsizeof(all_b_shares[clients[user_id].ftsa_uid]) + getsizeof(encrypted_zero_shares[clients[user_id].ftsa_uid]) for user_id in participating_ids]
    
        sec_agg_server_time -= time.time()
        total_samples_per_class = self.ftsa_server.online_construct(all_b_shares, encrypted_zero_shares.values())
        sec_agg_server_time += time.time()
        
        total_data_transfer = sum(enc_round_client_msg_sizes) + sum(enc_round_server_msg_sizes) + sum(agg_round_client_msg_size) + len(participating_ids) * getsizeof(total_samples_per_class)
        
        print(f'SecAgg of total samples per labels (server): {sec_agg_server_time * 1000:.0f} ms')
        print(f'SecAgg of total samples per labels (avg per client) {sec_agg_client_time * 1000 / len(participating_ids):.0f} ms')
        print(f'Total data transfers: {total_data_transfer} bytes (total), {total_data_transfer / len(participating_ids):.0f} bytes (per client)')
    
        return total_samples_per_class, sec_agg_server_time, (sec_agg_client_time / len(participating_ids)), total_data_transfer, total_data_transfer / len(participating_ids)
    
    def aggregate_model_updates(self, clients: Mapping[int, Client], participating_ids: List[int], alive_ids: List[int], global_model: Mapping[str, Any]):
        aggregated_model_updates = copy.deepcopy(global_model)
        if self.args.agg_type == 'fedavg':
            weights = {}
            if self.args.fedavg_type == 'weighted':
                total_samples = 0
                for user_id in alive_ids:
                    total_samples += self.total_samples_per_user[user_id]
                
                for user_id in alive_ids:
                    weights[user_id] = self.total_samples_per_user[user_id] / total_samples
            elif self.args.fedavg_type == 'unweighted':
                total_users = len(alive_ids)
                for user_id in alive_ids:
                    weights[user_id] = 1 / total_users

            for i, user_id in enumerate(alive_ids):
                if i == len(alive_ids) - 1:
                    print(f'{user_id}: {weights[user_id]:.4f}')
                else:
                    print(f'{user_id}: {weights[user_id]:.4f}', end=', ')
                
            for params in aggregated_model_updates.keys():
                aggregated_model_updates[params] = 0
            for user_id in tqdm(alive_ids, desc='Aggregating model updates'):
                model_update = clients[user_id].get_model_update()
                for params in aggregated_model_updates.keys():
                    aggregated_model_updates[params] += model_update[params] * weights[user_id]
        
        elif self.args.agg_type == 'fedla':
            if not self.args.sec_agg:
                total_samples_per_class = [0] * self.num_classes
                for label in range(self.num_classes):
                    for user_id in alive_ids:
                        total_samples_per_class[label] += self.total_samples_per_class_per_user[user_id][label]
                
                weights = {}
                for user_id in alive_ids:
                    user_power = {}
                    for label in range(self.num_classes):
                        # W(c_i, l_j) = S(c_i, l_j) / S(l_j)
                        if total_samples_per_class[label] != 0:
                            user_power[label] = self.total_samples_per_class_per_user[user_id][label] / total_samples_per_class[label]
                        else:
                            user_power[label] = 0
                    
                    # W(c_i) = SUM(W(c_i, l_j))
                    user_power = sum(user_power.values())
                    total_power = len(total_samples_per_class) - total_samples_per_class.count(0)
                    # W_FedLA(c_i) = W(c_i) / SUM(W(c_i))
                    weights[user_id] = user_power / total_power
                
                for i, user_id in enumerate(alive_ids):
                    if i == len(alive_ids) - 1:
                        print(f'{user_id}: {weights[user_id]:.4f}')
                    else:
                        print(f'{user_id}: {weights[user_id]:.4f}', end=', ')
                
                for params in aggregated_model_updates.keys():
                    aggregated_model_updates[params] = 0
                for user_id in tqdm(alive_ids, desc='Aggregating model updates'):
                    model_update = clients[user_id].get_model_update()
                    for params in aggregated_model_updates.keys():
                        aggregated_model_updates[params] += model_update[params] * weights[user_id]
            
            else:
                for i, user_id in enumerate(alive_ids):
                    if i == len(alive_ids) - 1:
                        print(f'{user_id}: {clients[user_id].get_weight():.4f}')
                    else:
                        print(f'{user_id}: {clients[user_id].get_weight():.4f}', end=', ')
                
                self.step += 1
                self.ftsa_server.new_fl_step(self.step)
                sec_agg_client_time = time.time()
                all_eb_shares, encrypted_quantized_model_updates = {}, {}
                for user_id in tqdm(alive_ids, desc='SecAgg (Online Encrypt)'):
                    eb_shares, encrypted_quantized_model_update = clients[user_id].get_model_update(self.step)
                    all_eb_shares[clients[user_id].ftsa_uid] = eb_shares
                    encrypted_quantized_model_updates[clients[user_id].ftsa_uid] = encrypted_quantized_model_update
                sec_agg_client_time = time.time() - sec_agg_client_time
                enc_round_client_msg_sizes = [getsizeof(all_eb_shares[clients[user_id].ftsa_uid]) + getsizeof(encrypted_quantized_model_updates[clients[user_id].ftsa_uid]) for user_id in alive_ids]
                
                sec_agg_server_time = time.time()
                all_eb_shares = self.ftsa_server.online_encrypt(all_eb_shares, encrypted_quantized_model_updates)
                sec_agg_server_time = time.time() - sec_agg_server_time
                enc_round_server_msg_sizes = [getsizeof(all_eb_shares[clients[user_id].ftsa_uid]) for user_id in alive_ids]
                
                if (len(alive_ids) < len(participating_ids) or len(alive_ids) < self.args.num_users) and self.args.use_approx_updates:
                    sec_agg_client_time -= time.time()
                    all_b_shares = {}
                    encrypted_dropout_shares = {}
                    for user_id in tqdm(alive_ids, desc='SecAgg (Online Construct)'):
                        b_shares, encrypted_dropout_share = clients[user_id].get_dropout_approx_share(all_eb_shares[clients[user_id].ftsa_uid], len(alive_ids), global_model)
                        all_b_shares[clients[user_id].ftsa_uid] = b_shares
                        encrypted_dropout_shares[clients[user_id].ftsa_uid] = encrypted_dropout_share
                    sec_agg_client_time += time.time()
                
                else:
                    sec_agg_client_time -= time.time()
                    all_b_shares = {}
                    encrypted_dropout_shares = {}
                    for user_id in tqdm(alive_ids, desc='SecAgg (Online Construct)'):
                        _, b_shares, encrypted_dropout_share = clients[user_id].ftsa_client.online_construct(all_eb_shares[clients[user_id].ftsa_uid])
                        all_b_shares[clients[user_id].ftsa_uid] = b_shares
                        encrypted_dropout_shares[clients[user_id].ftsa_uid] = encrypted_dropout_share
                    sec_agg_client_time += time.time()
                
                agg_round_client_msg_size = [getsizeof(all_b_shares[clients[user_id].ftsa_uid]) + getsizeof(encrypted_dropout_shares[clients[user_id].ftsa_uid]) for user_id in alive_ids]
                    
                sec_agg_server_time -= time.time()
                aggregated_quantized_model_updates = self.ftsa_server.online_construct(all_b_shares, encrypted_dropout_shares.values())
                dequantized_aggregated_model_updates = dequantize_vector(aggregated_quantized_model_updates, CLIPPING_RANGE, TARGET_RANGE)
                if (len(alive_ids) < len(participating_ids) or len(alive_ids) < self.args.num_users) and self.args.use_approx_updates:
                    dequantized_aggregated_model_updates -= (2 * len(alive_ids) - 1) * CLIPPING_RANGE
                else:
                    dequantized_aggregated_model_updates -= (len(alive_ids) - 1) * CLIPPING_RANGE
                aggregated_model_updates = reshape_back_to_dict(dequantized_aggregated_model_updates, aggregated_model_updates)
                sec_agg_server_time += time.time()
                
                total_data_transfer = sum(enc_round_client_msg_sizes) + sum(enc_round_server_msg_sizes) + sum(agg_round_client_msg_size) + len(alive_ids) * getsizeof(aggregated_model_updates)
                
                print(f'SecAgg of model updates (server): {sec_agg_server_time * 1000:.0f} ms')
                print(f'SecAgg of model updates (avg per client): {sec_agg_client_time * 1000 / len(alive_ids):.0f} ms')
                print(f'Total data transfers: {total_data_transfer} bytes (total), {total_data_transfer / len(alive_ids):.0f} bytes (per client)')
        if self.args.agg_type == 'fedla' and self.args.sec_agg:
            return aggregated_model_updates, sec_agg_server_time, (sec_agg_client_time / len(alive_ids)), total_data_transfer, total_data_transfer / len(alive_ids)
        else:
            return aggregated_model_updates
    
    def test_inference(self, global_model: Mapping[str, Any]):
        """Evaluates the model on a provided test dataset and calculates accuracy and loss.
        
        Args:
        - args: A set of arguments specifying configuration parameters.
        - model: The trained model to evaluate.
        - test_dataset: The dataset to test the model on.
        
        Returns:
        - A tuple containing accuracy and total loss on the test dataset.
        """
        
        self.model.load_state_dict(global_model)
        self.model.eval() # Set model to evaluation mode
        loss, total, correct = 0.0, 0.0, 0.0 # Initialize metrics

        # Evaluate the model on the test dataset
        for _, (images, labels) in enumerate(self.__test_loader):
            images, labels = images.to(self.device), labels.to(self.device) # Move data to device

            outputs = self.model(images) # Forward pass
            batch_loss = self.criterion(outputs, labels) # Compute loss
            loss += batch_loss.item() # Accumulate loss

            # Get predictions and calculate accuracy
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1) # Flatten predictions
            correct += torch.sum(torch.eq(pred_labels, labels)).item() # Count correct predictions
            total += len(labels) # Total number of labels processed

        accuracy = correct / total # Calculate accuracy
        return accuracy, loss # Return accuracy and loss