
import glob, os, sys, contextlib, re
# import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from torch.utils.data import WeightedRandomSampler
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
import matplotlib.pyplot as plt

import mlflow

from pydub import AudioSegment
import wave, librosa
print(f"Running in {os.getcwd()}")
print("External packages imported\n")

# Select where you are running this script -----------------#

google = False
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
    dataPath = rootPath + '/processed-data'


else:
    rootPath = './speechRecForMeeting'
    dataPath = './processed-data'

# -----------------------------------------------------------

from data_preprocessing import *
print('Running in ', os.getcwd())

# # DagsHub set-up --------------------------------
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'team-token'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f01653d37636d9488c48cd922f6ab83881d2cf4a'
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = 'speechRecForMeeting'

# mlflow.set_tracking_uri(f'https://dagshub.com/Viv-Crowe/speechRecForMeeting.mlflow')

sys.path.append(rootPath + '/py files')
from data_preprocessing import *
from data_loader import *
from CNN import *

## Data pre-processing ##

with open(dataPath + '/dialogue-acts-prepped.pkl','rb') as f:
    df_diag_acts = pickle.load(f)

segment_full_paths, df_timestamps, p = processSignals(dataPath + "/Signals-1")
prepareDataset(segment_paths, df_diag_acts, df_timestamps, p)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

## Load dataset ##
pickle_file = dataPath + "/dataset-5.pkl"
train_dataloader, val_dataloader, test_dataloader, p = prepareData(pickle_file)
examineBatches(train_dataloader, val_dataloader, test_dataloader)
print("Parameters: ", p)

exp_id_10M = mlflow.create_experiment("10 meetings")

## Train dataset ##
CNN = initialize()
criterion = nn.CrossEntropyLoss()
p['lr'] = 0.01
p['momentum'] = 0.9
p['weight_decay'] = 
optimizer = optim.SGD(CNN.parameters(), lr=p['lr'], momentum=p['momentum'])

with mlflow.start_run(run_name="Testing new metrics", experiment_id=exp_id_10M):
    use_gpu = torch.cuda.is_available()
    p['num epochs'] = 5
    tr = train(CNN, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=p['num epochs'], use_gpu=use_gpu)
    p['total_params'] = sum(p.numel() for p in CNN.parameters())
    train_error_rates, train_losses, test_error_rates, test_losses = tr

    y_hat, y_true, losses, error = prediction(test_dataloader, CNN, criterion)

    m = {}
    m['roc_auc'], precision, recall, m['accuracy'], y_hat_class = evaluate(y_hat, y_true)
    m['pr_auc'] = auc(precision, recall)
    # Log parameters + metrics

    mlflow.log_params(p)
    mlflow.log_metrics(m)

    fig = plt.plot(precision, recall)
    mlflow.log_figure(fig, "pr_curve.png")

    for s in range(p['num epochs']):
        mlflow.log_metric('train error', train_error_rates[s], s)
        mlflow.log_metric('train loss', train_losses[s], s)
        mlflow.log_metric('val error', train_error_rates[s], s)
        mlflow.log_metric('val loss', train_losses[s], s)

    
    # for i in epoch:
    #   mlflow.log_metrics(test_error, step=i)
    

