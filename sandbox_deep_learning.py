import os
import pandas as pd
import numpy as np
import scipy as sp
import librosa
import sys
import json
import itertools
import matplotlib.pyplot as plt
import re

import glob
from bs4 import BeautifulSoup
import importlib
# sys.path.append(os.path.expanduser("~/Dropbox/Ircam/code"))
import segmenter

# Build inverted list to match annotations to audio files
json_files = glob.glob(os.path.expanduser("~/Documents/repositories/JAAH/annotations/*.json"))
html_files = glob.glob(os.path.expanduser("~/Documents/repositories/JAAH/docs/data/*.html"))
audio_files = glob.glob(os.path.expanduser("~/Documents/data/JAAH/*/*/*.flac"))
json_stems = [os.path.splitext(os.path.basename(fn))[0] for fn in json_files]
html_stems = [os.path.splitext(os.path.basename(fn))[0] for fn in html_files]

# Build line of database:
def link_audio_paths():
	global html_slides
	jaah_info = pd.DataFrame(data=html_stems,columns=["stem"])
	jaah_info['audio_path'] = 0
	jaah_info['audio_exists'] = 0
	for id in jaah_info.index:
		fn = os.path.expanduser("~/Documents/repositories/JAAH/docs/data/") + jaah_info.loc[id]['stem'] + ".html"
		with open(fn) as fp:
			soup = BeautifulSoup(fp)
		spans = soup.find_all('span')
		targets = [span.text for span in spans if "$JAZZ_HARMONY_DATA_ROOT" in span.text]
		clean_target = targets[0].replace("$JAZZ_HARMONY_DATA_ROOT/",os.path.expanduser("~/Documents/data/JAAH/"))[1:-1]
		jaah_info['audio_path'][id] = clean_target
		jaah_info['audio_exists'][id] = os.path.exists(clean_target)
	jaah_info.to_csv("audio_paths_index.csv")

def boost_index_with_song_info():
	jaah_info = pd.read_csv("audio_paths_index.csv")
	# We say "target[0]" because there are instances where there are multiple instances of JAZZ_HARMONY_

def parse_annotation(ind):
	jaah_info = pd.read_csv("audio_paths_index.csv")
	annotation_path = os.path.expanduser("~/Documents/repositories/JAAH/annotations/{0}.json".format(jaah_info.loc[ind]["stem"]))
	with open(annotation_path, 'rb') as f:
		ann_json = json.load(f)
	audio_path = jaah_info['audio_path'][ind]
	part_names = [part['name'] for part in ann_json['parts']]
	part_types = [re.split("(?:- )|(?: -)",pn)[:1] for pn in part_names]
	part_forms = [re.split("(?:- )|(?: -)",pn)[1:] for pn in part_names]
	part_onsets = [part['beats'][0] for part in ann_json['parts']]
	all_onsets = [part['beats'] for part in ann_json['parts']]
	# all_onsets = list(itertools.chain.from_iterable(all_onsets))
	return ann_json, part_names, part_types, part_forms, part_onsets, all_onsets

jaah_info = pd.read_csv("audio_paths_index.csv")


import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation

# Deep Learning

import keras
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

def load_spectrograms(spectrogram_filename):
	# load spectrograms
	with np.load(spectrogram_filename) as npz:
		spectrograms = npz["features"]
		spec_clip_ids = npz["clip_id"]
	# create dataframe that associates the index order of the spectrograms with the clip_ids
	spectrograms_clip_ids = pd.DataFrame({"spec_id": np.arange(spectrograms.shape[0])}, index = spec_clip_ids)
	spectrograms_clip_ids.index.name = 'clip_id'
	return spectrograms, spectrograms_clip_ids

def standardize(data):
	# vectorize before standardization (cause scaler can't do it in that format)
	N, ydim, xdim = data.shape
	data = data.reshape(N, xdim*ydim)
	# standardize
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	# reshape to original shape
	return data.reshape(N, ydim, xdim)

def add_channel(data, n_channels=1):
	# n_channels: 1 for grey-scale, 3 for RGB, but usually already present in the data
	N, ydim, xdim = data.shape
	if keras.backend.image_data_format() == 'channels_last':  # TENSORFLOW
		# Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
		data = data.reshape(N, ydim, xdim, n_channels)
	else: # THEANO
		# Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
		data = data.reshape(N, n_channels, ydim, xdim)		
	return data


import os
from os.path import join
base_path = os.path.expanduser("~/Documents/repositories/ismir2018_tutorial/")
SPECTROGRAM_PATH = base_path + "ISMIR2018_tut_melspecs_subset"
METADATA_PATH =  base_path + "metadata"
# here, %s will be replace by 'instrumental', 'genres' or 'moods'
LABEL_FILE_PATTERN = join(METADATA_PATH, 'ismir2018_tut_part_1_%s_labels_subset_w_clipid.csv') 
SPECTROGRAM_FILE_PATTERN = join(SPECTROGRAM_PATH, 'ISMIR2018_tut_melspecs_part_1_%s_subset.npz')

# Collect some mel spectrograms? It's anot a state-of-the-art approach, but oh well.

# First, 









