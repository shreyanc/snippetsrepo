import csv
import os
import random
from typing import Any
from time import time
import numpy as np
import torch

class Profiler:
    """From https://github.com/acids-ircam/RAVE/blob/master/rave/model.py"""
    def __init__(self):
        # Initialize the list of ticks with a single item representing the start time
        self.ticks = [[time(), None]]

    def tick(self, msg):
        # Mark a specific point in the code for profiling
        # Add a new item to the list of ticks with the current time and the provided message
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"  # Separator line for the report
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"  # Append the message and elapsed time to the report
        rep += 80 * "=" + "\n\n\n"  # Separator line for the end of the report
        return rep



class Meters:
    def __init__(self, variable_names=None):
        self.variables = {}
        self.history = {}
        
        if variable_names is not None:
            for name in variable_names:
                self.register(name)

    def register(self, name):
        self.variables[name] = []
        self.history[name] = []

    def update(self, data, step=None):
        for name, value in data.items():
            if name in self.variables:
                self.variables[name].append(value)
                self.history[name].append((len(self.variables[name]) if not step else step, value))

    def history(self, name):
        if name in self.history:
            return self.history[name]

    def mean(self, name):
        if name in self.variables:
            values = self.variables[name]
            if len(values) > 0:
                return sum(values) / len(values)

    def recent_value(self, name):
        if name in self.variables:
            values = self.variables[name]
            if len(values) > 0:
                return values[-1]
            else:
                return 0.0

    def clear_history(self, name):
        if name in self.history:
            self.history[name] = []

    def get_recent_values(self, decimals=4):
        recent_values = {}
        for name in self.variables:
            recent_values[name] = np.round(self.recent_value(name), decimals)
        return recent_values
    
    def export_all_history(self):
        for name in self.variables:
            self._export_single_history(name)

    def _export_single_history(self, variable_name):
        if variable_name in self.variables:
            filename = f'history_{variable_name}.csv'
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', variable_name])
                hist = self.history[variable_name]
                for index, value in hist:
                    writer.writerow([index, value])

class EasyDict(dict):
    """
    Allows you to access and modify dictionary keys using dot notation instead of the usual square bracket notation.
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def num_if_possible(s):
    try:
        return int(s)
    except Exception as e:
        pass

    try:
        return float(s)
    except Exception as e:
        pass

    if s in ['True', 'true']:
        return True
    if s in ['False', 'false']:
        return False

    return s

def list_files_deep(dir_path, full_paths=True, filter_ext=None):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dir_path, '')):
        if len(filenames) > 0:
            for f in filenames:
                if full_paths:
                    all_files.append(os.path.join(dirpath, f))
                else:
                    all_files.append(f)

    if filter_ext is not None:
        return [f for f in all_files if os.path.splitext(f)[1] in filter_ext]
    else:
        return all_files


def inf(dl):
    """Infinite dataloader"""
    while True:
        for x in iter(dl): yield x


def choose_rand_index(arr, num_samples):
    return np.random.choice(arr.shape[0], num_samples, replace=False)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pytorch_random_sampler(dset, num_samples):
    assert num_samples < len(dset)
    sample_indices = np.random.choice(len(dset), num_samples)
    return Subset(dset, sample_indices)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir='.', saved_model_name="model_chkpt",
                 condition='minimize'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saved_model_name = saved_model_name
        self.save_path = os.path.join(self.save_dir, self.saved_model_name + '.pt')
        self.condition = condition
        assert condition in ['maximize', 'minimize']
        self.metric_best = np.Inf if condition == 'minimize' else -np.Inf

    def __call__(self, metric, model):

        score = metric if self.condition == 'maximize' else -metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Metric improved ({self.condition}) ({self.metric_best:.6f} --> {metric:.6f}).  Saving model to {os.path.join(self.save_dir, self.saved_model_name + ".pt")}')
        torch.save(model.state_dict(), self.save_path)
        self.metric_best = metric
