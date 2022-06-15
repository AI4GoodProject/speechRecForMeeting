
import glob, os, sys, contextlib, re
import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
# import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from collections import Counter

from pydub import AudioSegment
import wave, librosa
print(f"Running in {os.getcwd()}")
print("External packages imported\n")

google = False
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
    dataPath = rootPath
else:
    rootPath = '/speechRecForMeeting'
    dataPath = ''
print(f"path to py files:{rootPath + '/data and code/py files'}")
sys.path.append(rootPath + '/py files')
import data_preprocessing 
# import logistic_model
if prepareDataset:
    print("Our scripts imported\n")
else:
    print("Not importing")

# Pre-processing

# [segment_full_paths, df_timestamps] = processSignals("Signals-10M", rootPath)
# segment_paths = glob.glob('./segments-5/*.wav')
# with open('./processed-data/dialogue-acts-prepped.pkl', "rb") as f:
#     df_diag_acts = pickle.load(f)
# p = {'segment_length': 10, 'overlap_length': 1}

prepareDataset(segment_paths, df_diag_acts, p)

# [features, df_timestamps] = processSegments("Signals-10M")
# diag_acts_path = processDialogueActs(path2all_xml_files)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

# from logistic_model import *

# [model, features] = initialize(features)

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

#     # # Balance dataset
#     count=Counter(y_train)
#     class_count=np.array([count[0],count[1]])
#     weight=1./class_count
#     print(weight)

#     samples_weight = np.array([weight[t] for t in y_train])
#     samples_weight = torch.from_numpy(samples_weight)

#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('/files/', train=True, download=True),
#     batch_size=batch_size_train, **kwargs, sampler = sampler)

# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('files/', train=False, download=True),
#     batch_size=batch_size, **kwargs)

# for i, (data, label) in enumerate(torch.utils.data.trainLoader):
#     count=Counter(label.np())
#     print("batch-{}, 0/1: {}/{}".format(i, count[0], count[1]))

# # # Train and evaluate model
# model = train(model, x_train, y_train)


# results = evaluate(model, x_test, y_test)

