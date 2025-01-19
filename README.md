# Source Code for *SAAFL: Secure Aggregation Approach for Label-Aware Federated Learning*

## Installing Dependencies
```bash
conda env create -f saafl.yml
```

## Running Experiments
1. Change to `src` directory.
```bash
cd src
```
2. Explore the experiment paramters.
```bash
python federated_main.py --help
```
3. Run an experiment.
```bash
python federated_main.py --dataset=emnist-balanced --epochs=100 --local_ep=10 --lr=0.01 --frac=0.7 --unequal --num_samples_per_user=2400 --seed=42 --num_unique_classes_non_iid=1 --num_non_iid_users=7 --agg_type=fedla --sec_agg --dropout=1 --first_dropout_epoch=1 --last_dropout_epoch=100 --use_approx_updates
```

The above command runs an experiment using SAAFL (`--agg_type=fedla`, `--sec_agg`, and `--use_approx_updates`) with `k_r = 0.7`, `unique_c = 1`, `noniid_s = 0.7`, and 1 dropped user from FL round 1 to 100.

## References
- [FTSA Implementation](https://github.com/MohamadMansouri/fault-tolerant-secure-agg).
M. Mansouri, M. Önen, W. Ben Jaballah, "Learning from Failures: Secure and Fault-Tolerant Aggregation for Federated Learning," in Proceedings of the 38th Annual Computer Security Applications Conference, 2022, pp. 146–158.
- [FedLA Implementation](https://github.com/AhmadMkhalil/Label-Aware-Aggregation-in-Federated-Learning).
Khalil, A., et al, "Label-Aware Aggregation for Improved Federated Learning," in 2023 Eighth International Conference on Fog and Mobile Edge Computing (FMEC), 2023, pp. 216-223.
- [Opacus Library](https://github.com/pytorch/opacus)