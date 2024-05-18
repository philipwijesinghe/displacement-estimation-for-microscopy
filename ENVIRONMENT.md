# Python Environment

Using Miniconda or Anaconda (conda package and environment management).

Code was tested with:  
- Windows 10
- NVidia GTX 1080Ti (GPU)
- PyTorch 2.2.1  
- CUDA 12.1 (nvidia driver >=528.33)
- Python 3.11

Please follow the most up-to-date instructions for installing (PyTorch)[https://pytorch.org/get-started/locally/].

Briefly:  
First install the most recent nvidia driver and verify CUDA is installed with cmd: ```nvidia-smi```

Create a new conda environment in Command Prompt  
```conda create -n <name>``` e.g., ```conda create -n deformnet```

Activate the environment  
```conda activate deformnet```

Install python and then pytorch  
```conda install python=3.11```
```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```

Test if CUDA is available in python, by running python in cmd  
```
python
```
then  
```python
>>> import torch
>>> torch.cuda.is_available()
``` 
should return ```true```


## Packages required for inference only

Install the following packages by executing in cmd:  
```
conda install <package-name>
```
Packages:  
```
scipy
tqdm
```

## Additional packages required for training

Packages:

```
joblib
torchmetrics
tensorboard
```

as well as:
```
pip3 install git+https://github.com/pvigier/perlin-numpy
```

Optionally, Weights and Biases (`wandb`) may be used for cloud-based training run tracking. To enable it, install the required package with
```
conda install wandb
```
and add
```
wandb:
  enable: True
  project: 'your_project_name'
  entity: 'your_entity_name'
```
to your training config.

