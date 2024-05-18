# Python Environment

For convinience, environment files for a `conda`-based package manager (Anaconda/miniconda/mamba/etc.) are provided. To set up for inference with `conda`:
```
conda env create -f environment-inference.yml
conda activate deformnet-inference
```

To set up for training using the CUDA backend (NVIDIA GPU):
```
conda env create -f environment-training-cuda.yml
conda activate deformnet
```

To set up for training using the MPS backend (Apple Silicon GPU):
```
conda env create -f environment-training-cuda.yml
conda activate deformnet
```

## Manual Installation

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


### Packages required for inference only

Install the following packages by executing in cmd:  
```
conda install <package-name>
```
Packages:  
```
scipy
tqdm
```

### Additional packages required for training

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
## Compatability

Our training code currently only supports the CPU and CUDA PyTorch backends. 

Code was tested with:  
- Windows 10
- NVidia GTX 1080Ti (GPU)
- PyTorch 2.2.1  
- CUDA 12.1 (nvidia driver >=528.33)
- Python 3.11
