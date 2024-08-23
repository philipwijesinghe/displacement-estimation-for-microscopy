# -*- coding: utf-8 -*-
""" Defines the Config Class for storing and setting DL parameters

This is an example of how to store, save and manage configuration parameters
"""

import copy
import os

import yaml

"""Default config dict

TODO: Populate dict with default values/params that you want to be modifiable
"""
_default_config = {
    'run_name': 'default',
    'group': 'default',
    'img_size': [256, 256],
    'model': 'UNet',
    'epochs': 300,
    'batch_size': 8,
    'lr': 1e-4,
    'b1': 0.9,
    'b2': 0.999,
    'dirs': {
        'train': '',
        'val': '',
        'output': ''
    },
    'loss': ['MultiscaleLoss'],
    'loss_args': [],
    'loss_weights': [],
    'use_rewarp': 0,
    'rewarp_weight': 1,
    'rewarp_loss': 'NPCC',
    'transforms': {
        'normalisation': {  # add kwargs to pass to function if needed
        },
        'noise': 0,
        'noise_gauss': 0,
        'blur_sigma': 0,
        'train': [
            'SimultaneousCrop'
        ],
        'val': [
            'SimultaneousCrop'
        ]
    },
    'scheduler': {
        'use': 0,
        'milestones': [40, 80, 120, 160, 200, 240],
        'gamma': 0.5
    },
    'update_period': 100,
    'save_period': 100,
    'wandb': {
        'enable': False,
        'project': 'example_project',
        'entity': 'example_entity',
    }
}


def default():
    """Returns a default config dict as a new instance

    :return:
    default config instance
    """
    return copy.deepcopy(_default_config)


def overwrite(old, new):
    """Overwrites a config with values from a new one if they exist

    :param old:
        config instance to be overwritten
    :param new:
        new config values
    """
    _recursive_merge_strings(old, new)
    return


def read_yaml(path_to_yml):
    """Reads raw values from .yml file into a dict

    :param path_to_yml:
        path to .yml file
    :return:
        dict
    """
    with open(path_to_yml, 'r') as stream:
        try:
            raw = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return raw


def load(path_to_yml, config=None):
    """Loads .yml file and populates a config dict

    :param path_to_yml:
        path to .yml file
    :param config:
        (Optional) config instance to overwrite with yml, otherwise uses a new default instance
    :return:
        config dict
    """
    if config is None:
        config = default()
    raw = read_yaml(path_to_yml)
    overwrite(config, raw)

    if not config['dirs']['output']:
        head = os.path.split(os.path.abspath(path_to_yml))
        config['dirs']['output'] = head[0]
    if not config['dirs']['data_root']:
        config['dirs']['data_root'] = config['dirs']['output']
    if not config['dirs']['train']:
        config['dirs']['train'] = os.path.join(config['dirs']['data_root'], 'train')
    if not config['dirs']['val']:
        config['dirs']['val'] = os.path.join(config['dirs']['data_root'], 'val')
        
    return config


def save(path_to_yml, config):
    """Saves config to file

    :param path_to_yml:
        path to .yml file
    :param config:
        config dict to save
    """
    with open(path_to_yml, 'w') as file:
        yaml.dump(config, file)


def _recursive_merge_strings(old, new):
    """Recursively merge the new dictionary into the config iterating
    through levels
    """
    for key, val in new.items():
        # Follow the default configuration template
        if key in old:
            # if nested
            if isinstance(old[key], dict):
                # go again
                _recursive_merge_strings(old[key], new[key])
            # maintain type
            elif isinstance(old[key], str):
                old[key] = str(new[key])
            elif isinstance(old[key], (int, float, complex)):
                if isinstance(new[key], (int, float, complex)):
                    old[key] = new[key]
                elif isinstance(new[key], str):
                    old[key] = eval(new[key])
            elif isinstance(old[key], tuple):
                if isinstance(new[key], (tuple, list)):
                    old[key] = tuple(new[key])
            # failsafe overwrite
            else:
                old[key] = new[key]
        else:
            old[key] = new[key]
