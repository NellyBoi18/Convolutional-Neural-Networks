# Convolutional Neural Networks
 Implement and train a convolutional neural network (CNN), specifically LeNet

# How to Set Up a Python Virtual Environment
**You will need to use Python 3 and a Python Virtual Environment with torch v1.12.1+cpu, torchvision v0.13.1+cpu, and torchaudio v0.12.1+cpu**

The following steps sets up a Python Virtual Environment using the venv module but you can use other virtual envs such as Conda.

**Step 1:** Set up a Python Virtual Environment named Pytorch:
```
python3 -m venv /path/to/new/virtual/environment
```
For example, if you want to put the virtual environment in your working directoy:
```
python3 -m venv Pytorch
```

**Step 2:** Activate the environment
```
source Pytorch/bin/activate
```

**Step 3:** Upgrade pip and install the CPU version of Pytorch. 
(Note: you may be using pip3 instead of pip so just add 3 if needed)
```
pip install --upgrade pip
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```


You can check the version of the packages installed using:
```
pip freeze
```
To deactive the virtual environment:
```
deactivate
```

# How To Run
**train_miniplaces.py**

The first time you run train_miniplaces.py, the data loader will try to download the full dataset. If this does not work, you may have to manually download the datasets. 

[Data](http://miniplaces.csail.mit.edu/data/data.tar.gz)

[Train](http://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt)

[Val](http://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt)

After you manually download these, move them into the ./data/miniplaces folder.

When you run train_miniplaces.py, the python script will save two files in the ./outputs folder.
- **checkpoint.pth.tar** is the model checkpoint at the latest epoch.
- **model_best.pth.tar** is the model weights that has highest accuracy on the validation set.

**Starting and Stopping Training:**

The code supports resuming from a previous checkpoint, such that you can pause the training and resume later.
This can be achieved by running
```
python train_miniplaces.py --resume ./outputs/checkpoint.pth.tar
```

**Evaluating The Model:**
After training, run eval_miniplaces.py to evaluate the model on the validation set and also help in timing the model. This script will grab a pre-trained model and evaluate it on the validation set of 10K images. For example, you can run
```
python eval_miniplaces.py --load ./outputs/model_best.pth.tar
```