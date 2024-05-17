# -*- coding: utf-8 -*-
""" Training script for deep learning displacement estimation

This script should be the entry point for the user to train a network (or several) based on a configuration file

The output folder should take the form
../example/models/example_model/config.yml
                               /<network training outputs>

OR
../example/models/example_model/config.yml
                               /subexperiment1/<network training outputs>
                               /subexperiment2/
                               ...

"""

import os
import deeplearning.config as config
from deeplearning.trainer import Trainer


""" USER CONFIG """
# Load a config that contains static parameters for the whole experiment
conf_paths = [
    r"C:\example\models\example_model_untrained\config.yml",
]
for conf_path in conf_paths:
    conf = config.load(conf_path)

    # Can train a single network
    dl = Trainer(conf)
    dl.train()

    # OR

    # Change some desired parameters and delegate to each a new folder
    # parameter_sweep = ['realEPE', 'SSIM_loss']
    #
    # run_name = conf['run_name']
    # out_dir = conf['dirs']['output']
    #
    # for i, p in enumerate(parameter_sweep):
    #     out_path = os.path.join(out_dir, '{0}'.format(p))
    #     conf['dirs']['output'] = out_path
    #     conf['run_name'] = run_name + '_{0}'.format(p)
    #     conf['loss'] = p
    #
    #     dl = Trainer(conf)
    #     dl.train()
