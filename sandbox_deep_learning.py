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
import jaah_experiment as jaah

# Build inverted list to match annotations to audio files
jaah_info = pd.read_csv("audio_paths_index.csv")
ann_json, part_names, part_types, part_forms, part_onsets, all_onsets, duration = jaah.parse_annotation(1)
importlib.reload(jaah)
info_df = jaah.parse_all_annotations()

song_id = 4
def load_audio_for_song_id(song_id):
	audio_path = jaah_info.audio_path.loc[song_id]
	y, sr = librosa.core.load(audio_path, sr=22050)
	return y, sr

def arrange_feat_images(feat, window_size_secs=3):
	# To take 5 second windows of MFCC feature frames, we need 5 seconds / (0.0929/4 sec/hop) = 216 frames
	feat_times = librosa.core.frames_to_time(range(feat.shape[1]))
	image_length = np.ceil(window_size_secs / (2048/22050/4)).astype(int)
	# Get windows by reshaping array 4 different times with different offsets, then unshuffling the arrays.
	n_images_with_offset = [np.floor((len(feat_times)-i*(1/4)*image_length)/image_length).astype(int) for i in range(4)]
	feat_image_sets = []
	feat_time_sets = []
	for i,n_frames in enumerate(n_images_with_offset):
		offset = (i*(1/4)*image_length).astype(int)
		feat_image_sets += [np.reshape(feat[:,offset:n_frames*image_length+offset], [feat.shape[0],n_frames,image_length])]
		feat_time_sets += [feat_times[np.arange(offset,len(feat_times),image_length)][:n_frames]]
	feat_image_times = np.concatenate(feat_time_sets)
	feat_images = np.concatenate(feat_image_sets,axis=1)[:,np.argsort(feat_image_times),:]
	return feat_images, feat_image_times


# window_size = 
# cqt, mfcc, rhyt, tempo, beat_inds, audio, sr = jaah.get_basic_feats(song_id)
# Note the hard-coded parameters, which are the librosa defaults:
mfsg = librosa.feature.melspectrogram(audio, n_fft=2048, hop_length=512)  # 0.0929 second windows with 1/4 window overlap

import os
my_base_path = os.path.expanduser("~/Documents/data/JAAH/feature_data/")
MY_MFCCS_PATH = my_base_path + "DTL_melspecs"
METADATA_PATH =  base_path + "metadata"
# here, %s will be replace by 'instrumental', 'genres' or 'moods'
LABEL_FILE_PATTERN = join(METADATA_PATH, 'ismir2018_tut_part_1_%s_labels_subset_w_clipid.csv') 
SPECTROGRAM_FILE_PATTERN = join(SPECTROGRAM_PATH, 'ISMIR2018_tut_melspecs_part_1_%s_subset.npz')

feature_file = 
with open(feature_file,'w') as file_handle:
	numpy.savez(file_handle, mfcc_images, mfcc_image_times)

def feature_vector(song_id, action='load',):
	assert action in ['load', 'save']
	if action=='save':
		

for i in range(4):
	plt.subplot(2,4,i+1)
	plt.imshow(mfcc_images[:,i,:],aspect='auto')
	plt.subplot(2,4,i+5)
	plt.imshow(mfcc[:,np.arange(np.int(i/4*image_length),np.int(i/4*image_length)+image_length)],aspect='auto')

plt.savefig("tmp.pdf")
# Check:
mfcc_image_set[:,0,:] == mfcc[:,:image_length]
a = np.reshape(np.arange(36),[6,6])
np.reshape(a,(3,3))


mfcc_images = np.reshape(mfcc,)

def compute_mfccs_for_song(audio_path):
	


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















# coding: utf-8

# # ISMIR 2018 Tutorial
# # Deep Learning for Music Information Retrieval
# 
# ## Part 1: Convolutional Neural Networks for Instrumental, Genre and Mood Recognition
# 
# Author: Thomas Lidy
# 
# This tutorial shows how different Convolutional Neural Network architectures are used for:
# * Instrumental vs. Vocal Detection:  detecting whether a piece of music is instrumental or contains vocals
# * Genre Classification
# * Mood Recognition
# 
# The data set used is a subset of the [MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) with only 1 sample excerpt of each of the original audio files.
# 
# The annotations of the original dataset contain a multitude of tags, which were preprocessed in Part 0 of this tutorial in order to create 3 groundtruth files for instrumental/vocal, genre and mood recognition.
# 
# Likewise, the original audio files were preprocessed to extract Mel spectrograms as an input for this Part 1 of the tutorial; also refer to Part 0 on how this preprocessing was done.

# ### Requirements
# 
# * Python >= 3.5
# * Keras >= 2.1.1
# * Tensorflow
# * scikit-learn >= 0.18
# * Pandas
# * Librosa
# * MatplotLib

# ### Download Data
# 
# If you haven't already (following the [README](./README.md#download-prepared-datasets)), 
# please download the following prepared data (from MagnaTagaTune data set) for this tutorial:
# 
# **Download prepared spectrograms:** https://owncloud.tuwien.ac.at/index.php/s/bxY87m3k4oMaoFl (96MB)
# 
# Unzip the file e.g. inside this Tutorial folder, and adapt the following `SPECTROGRAM_PATH` variable:

# In[1]:


# SET PATH OF DOWNLOADED DATA HERE
# (can be relative path if you unzipped the files inside this tutorial's folder)




# In[2]:


# IF YOU USE A GPU, you may set which GPU(s) to use here:
# (this has to be set before the import of Keras and Tensorflow)
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0,1,2,3" 


# In[3]:


# General Imports



# # 1) Instrumental vs. Vocal Detection
# 
# This is a binary classification task to detect whether a piece of audio is instrumental or vocal (= singing or voice). The output decision is *either* 0 *or* 1.

# ## Load Audio Spectrograms
# 
# We have pre-processed the audio files already and extracted Mel spectrograms. We load these from a Numpy .npz file, which contains the spectrograms and also the associated clip ids.

# In[72]:


task = 'instrumental'
SPECTROGRAM_FILE = SPECTROGRAM_FILE_PATTERN % task

with np.load(SPECTROGRAM_FILE) as npz:
    spectrograms = npz["features"]
    spec_clip_ids = npz["clip_id"]

# check how many spectrograms we have and their dimensions
spectrograms.shape


# In[73]:


import matplotlib.pyplot as plt
plt.imshow(spectrograms[100,:,:])
plt.show()
np.min(spectrograms[100,:,:])


# In[74]:


# double-check whether we have the same number of ids from spectrogram file
len(spec_clip_ids)


# In[75]:


# create dataframe that associates the index order of the spectrograms with the clip_ids
spectrograms_clip_ids = pd.DataFrame({"spec_id": np.arange(spectrograms.shape[0])}, index = spec_clip_ids)
spectrograms_clip_ids.index.name = 'clip_id'
spectrograms_clip_ids.head()


# In[76]:


# we define the same in a convenience function used later


# ### Show Mel Spectrogram (1 example just for illustration)

# In[77]:


# you can skip this if you do not have matplotlib installed

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


# take first spectrogram as an example
i = 100
spec = spectrograms[i]
np.max(spec), np.min(spec)


# In[79]:


# plot it 
fig = plt.imshow(spec, origin='lower', aspect='auto')
fig.set_cmap('jet')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


# ## Standardization
# 
# <b>Always standardize</b> the data before feeding it into the Neural Network! (unless you use BatchNormalization in your Neural Network)
# 
# We use <b>Zero-mean Unit-variance standardization</b> (also known as Z-score normalization).
# Here, we use <b>attribute-wise standardization</b>, i.e. each pixel is standardized individually, as opposed to computing a single mean and single standard deviation of all values.
# 
# ('Flat' standardization would also be possible, but we have seen benefits of attribute-wise standardization in our experiments).
# 
# We use the StandardScaler from the scikit-learn package for our purpose.
# As it works typically on vector data, we have to vectorize (i.e. reshape) our matrices first, and then reshape again to the original shape. We created a convenience function for that:

# In[80]:



# TODO: normalizing time windows doesn't make sense to me because there could be arbitrary time shifts.
# Scaling to frequency bins does make some sense though.
# So maybe collect all the bins for a given frequency (across the dataset) and normalize that as a single attribute?


# In[81]:


spectrograms = standardize(spectrograms)
print(np.min(spectrograms))
spectrograms.shape # verify that the shape is again the same as before


# ## Load the Metadata

# In[83]:


# use META_FILE_PATTERN to load the correct metadata file. set correct METADATA_PATH above
task = 'instrumental'
csv_file = LABEL_FILE_PATTERN % task

metadata = pd.read_csv(csv_file, index_col=0) #, sep='\t')
metadata.shape


# In[84]:


metadata.head()


# In[85]:


# how many instrumental tracks
metadata.sum()


# In[86]:


# how many vocal tracks
(1-metadata).sum()


# In[87]:


# baseline:
1260/len(metadata)


# ## Align Metadata and Spectrograms

# In[88]:


len(metadata)


# In[89]:


# check if we find all metadata clip ids in our spectrogram data
len(set(metadata.index).intersection(set(spec_clip_ids)))


# In[90]:


# we may have more spectrograms than metadata
spectrograms.shape


# ### Create Train X and Y: data and classes

# **get the correct spectrogram indices given the metadata's clip_ids in a sorted way**

# In[94]:


meta_clip_ids = metadata.index
spec_indices = spectrograms_clip_ids.loc[meta_clip_ids]['spec_id']


# **and select a correctly sorted subset of the original spectrograms for this task**

# In[95]:


data = spectrograms[spec_indices,:]
data.shape


# In[96]:


# for training convert from Pandas DataFrame to numpy array
classes = metadata.values
classes


# In[97]:


# number of classes is number of columns in metaddata
n_classes = metadata.shape[1]


# # Convolutional Neural Networks
# 
# A Convolutional Neural Network (ConvNet or CNN) is a type of (Deep) Neural Network that is well-suited for 2D axes data, such as images or spectrograms, as it is optimized for learning from spatial proximity. Its core elements are 2D filter kernels which essentially learn the weights of the Neural Network, and down-scaling functions such as Max Pooling.
# 
# A CNN can have one or more Convolution layers, each of them having an arbitrary number of N filters (which define the depth of the CNN layer), typically followed by a pooling step, which aggregates neighboring pixels together and thus reduces the image resolution by retaining only the average or maximum values of neighboring pixels.

# ## Preparing the Data
# 
# ### Adding the channel
# 
# As CNNs were initially made for image data, we need to add a dimension for the color channel to the data. RGB images typically have a 3rd dimension with the color. 
# 
# <b>Spectrograms, however, are considered like greyscale images, as in the previous tutorial.
# Likewise we need to add an extra dimension for compatibility with the CNN implementation.</b>
# 
# For greyscale images, we add the number 1 as the depth of the additional dimension of the input shape (for RGB color images, the number of channels is 3).

# In[100]:

keras.backend.image_data_format()


# In[101]:


data.shape


# In[102]:


data = add_channel(data, n_channels=1)
data.shape


# In[103]:


# we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of files)
input_shape = data.shape[1:]  
input_shape


# ### Train & Test Set Split
# 
# We split the original full data set into two parts: Train Set (75%) and Test Set (25%).
# 
# Note: 
# For demo purposes we use only 1 split here. A better way to do it is to use **Cross-Validation**, doing the split multiple times, iterating training and testing over the splits and averaging the results.

# In[104]:


# use 75% of data for train, 25% for test set
testset_size = 0.25


# In[147]:


# Stratified Split retains the class balance in both sets

splitter = StratifiedShuffleSplit(n_splits=0, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes)

for train_index, test_index in splits:
    #print("TRAIN INDEX:", train_index)
    #print("TEST INDEX:", test_index)
    #print("# of instances TRAIN:", len(train_index))
    #print("# of instances TEST:", len(test_index))
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes[train_index]
    test_classes = classes[test_index]
# Note: this for loop is only executed once if n_splits==1


# In[149]:


print(train_set.shape)
print(test_set.shape)


# # Creating Neural Network Models in Keras
# 
# ## Sequential Models
# 
# In Keras, one can choose between a Sequential model and a Graph model. Sequential models are simple concatenations of layers. Graph models can also handle those but also more complex neural network architectures. Keras now recommends to use the Graph models by default, but for a simple entry into the topic we are going to start with Sequential models first:

# **Exercise:** Try different configurations by uncommenting various lines of code in the following code box:
# * 1 Layer CNN
# * add 2nd Layer
# * increase number of conv_filters
# * add Dropout
# 
# Observe how the number of parameters in the model changes, and also the speed of training.

# In[181]:


#np.random.seed(0) # make results repeatable

model = Sequential()

# conv_filters = 16   # number of convolution filters (= CNN depth)
# UNCOMMENT TO INCREASE FILTERS
conv_filters = 32   # number of convolution filters (= CNN depth)

# 1st Layer
model.add(Convolution2D(conv_filters, (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2))) 

# # UNCOMMENT TO ADD 2nd LAYEER
model.add(Convolution2D(conv_filters, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2))) 

# UNCOMMENT TO ADD DROPOUT
model.add(Dropout(rate=0.25))
# rate = (1 - keep_prob)

# After Convolution, we have a conv_filters*y*x matrix output
# In order to feed this to a Full (Dense) layer, we need to flatten all data
# Note: Keras does automatic shape inference, i.e. it knows how many (flat) input units the next layer will need,
# so no parameter is needed for the Flatten() layer.
model.add(Flatten()) 

# Full layer
model.add(Dense(256, activation='sigmoid')) 

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(n_classes,activation='sigmoid'))


# **Model.summary() gives a nice overview of the model architecture and the number of weights (parameters) in the NN**

# In[182]:


model.summary()


# ## Training the CNN

# We have to define:
# 
# * loss function: binary crossentropy for binary or multi-label problems, categorical crossentropy for single class problems (custom loss functions are also possible)
# * optimizer: classic Stochastic Gradient Descent, or derivations thereof (e.g. Adam, ...)
# * metric: one or multiple metrics for evaluation on the train, validation and test sets
# * epochs: number of iterations to train the network (in the default case, in each epoch the full dataset is presented once to the network)
# * batch_size: how many instances are presented as one batch to the network, before a weight update (= Back Propagation) takes place

# In[183]:


# Define a loss function 
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd' 

# Which metric to evaluate
metrics = ['accuracy']

# Batch size
batch_size = 32

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# In[184]:


# TRAINING the model
# (execute multiple times to train more epochs)
epochs = 10
history = model.fit(train_set, train_classes, batch_size=batch_size, epochs=epochs)


# ### Verifying Accuracy on Test Set

# In[187]:


# always execute this, and then one of the boxes of accuracy_score below to print the result
test_pred = model.predict_classes(test_set)
# Note: we use model.predict_classes (only available in the Sequential model) to already round the prediction value to 0 or 1
# model.predict(test_set) gives you the raw values
#test_pred = model.predict(test_set)


# In[188]:


# show first 10 predictions
test_pred[0:10]


# In[179]:


# 1 layer
accuracy_score(test_classes, test_pred)


# In[180]:


# 2 layers
accuracy_score(test_classes, test_pred)


# In[189]:


# 2 layers + 32 convolution filters
accuracy_score(test_classes, test_pred)


# In[64]:


# 2 layer + 32 convolution filters + Dropout
accuracy_score(test_classes, test_pred)


# ## Additional Parameters & Techniques
# 
# **Exercise:** Try out more parameters and techniques: comment/uncomment appropriate lines of code below:
# * add ReLU activation
# * add Batch normalization
# * add Dropout on multiple layers

# In[207]:


model = Sequential()

conv_filters = 16   # number of convolution filters (= CNN depth)
filter_size = (3,3)
pool_size = (2,2)

# Layer 1
model.add(Convolution2D(conv_filters, filter_size, padding='valid', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=pool_size)) 
# model.add(Dropout(0.3))

# Layer 2
model.add(Convolution2D(conv_filters, filter_size, padding='valid', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=pool_size)) 
# model.add(Dropout(0.1))

# In order to feed this to a Full(Dense) layer, we need to flatten all data
model.add(Flatten()) 

# Full layer
model.add(Dense(256))  
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(n_classes,activation='sigmoid'))


# In[208]:


model.summary()


# In[209]:


# Compile the model
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# In[210]:


# Train the model
epochs = 10
history = model.fit(train_set, train_classes, batch_size=32, epochs=epochs)


# In[211]:


# Verify Accuracy on Test Set
test_pred = model.predict_classes(test_set)
accuracy_score(test_classes, test_pred)


# # 2) Genre Classification
# 
# In this Genre classification task, we have multiple classes, but the decision has to be made for 1 target class.
# This is called a single-label / multi-class task (as opposed to a multi-label task).

# ## Load Audio Spectrograms
# 
# We prepared already the Mel spectrograms for the audio files used in this task.

# In[70]:


task = 'genres'

# load Mel spectrograms
spectrogram_file = SPECTROGRAM_FILE_PATTERN % task
spectrograms, spectrograms_clip_ids = load_spectrograms(spectrogram_file)

# standardize
data = standardize(spectrograms)
data.shape # verify the shape of the loaded & standardize spectrograms


# ## Load Metadata

# In[71]:


# use META_FILE_PATTERN to load the correct metadata file. set correct METADATA_PATH above
csv_file = LABEL_FILE_PATTERN % task
metadata = pd.read_csv(csv_file, index_col=0) #, sep='\t')
metadata.shape


# In[72]:


metadata.head()


# In[73]:


# how many tracks per genre
metadata.sum()


# #### Baseline:
# 
# A 'dumb' classifier could assign all predictions to the biggest class. The number of tracks belonging to the biggest class divided by the total number of tracks in the dataset is our baseline accuracy in %.

# In[74]:


# baseline: 
metadata.sum().max() / len(metadata)


# ### Align Metadata and Spectrograms

# In[75]:


# check if we find all metadata clip ids in our spectrogram data
len(set(metadata.index).intersection(set(spectrograms_clip_ids)))


# In[76]:


spec_indices = spectrograms_clip_ids.loc[metadata.index]['spec_id']
data = spectrograms[spec_indices,:]
data.shape


# ### Create Train X and Y: data and classes

# In[77]:


# classes needs to be a "1-hot encoded" numpy array (which our groundtruth already is! we just convert pandas to numpy)
classes = metadata.values
classes


# In[78]:


n_classes = metadata.shape[1]


# In[79]:


# add channel (see above)
data = add_channel(data)
data.shape


# In[80]:


# input_shape: we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of files)
input_shape = data.shape[1:]  
input_shape


# ### Train & Test Set Split
# 
# We split the original full data set into two parts: Train Set (75%) and Test Set (25%).

# In[81]:


testset_size = 0.25 # % portion of whole data set to keep for testing, i.e. 75% is used for training


# In[82]:


# Stratified Split retains the class balance in both sets

splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes)

for train_index, test_index in splits:
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes[train_index]
    test_classes = classes[test_index]
# Note: this for loop is only executed once if n_splits==1


# In[83]:


print(train_set.shape)
print(test_set.shape)


# ## Model: Compact CNN
# 
# This is a 5 layer Convolutional Neural Network inspired and adapted from Keunwoo Choi (https://github.com/keunwoochoi/music-auto_tagging-keras)
# 
# * It is specified using Keras' functional Model **Graph API** (https://keras.io/models/model/).
# * It allows to specify 3, 4 or 5 Convolutional Layers.
# * It adapts the Pooling sizes according to the number of Mel bands use in the input.
# * It uses Batch Normalization.

# In[84]:


def CompactCNN(input_shape, nb_conv, nb_filters, normalize, nb_hidden, dense_units, 
               output_shape, activation, dropout, multiple_segments=False, input_tensor=None):
    
    melgram_input = Input(shape=input_shape)
    
    n_mels = input_shape[0]

    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 72:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 64:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]

    # Determine input axis
    if keras.backend.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2
            
    # Input block
    #x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)
        
    if normalize == 'batch':
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)
    elif normalize in ('data_sample', 'time', 'freq', 'channel'):
        x = Normalization2D(normalize, name='nomalization')(melgram_input)
    elif normalize in ('no', 'False'):
        x = melgram_input

    # Conv block 1
    x = Convolution2D(nb_filters[0], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[0], name='pool1')(x)
        
    # Conv block 2
    x = Convolution2D(nb_filters[1], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[1], name='pool2')(x)
        
    # Conv block 3
    x = Convolution2D(nb_filters[2], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[2], name='pool3')(x)
    
    # Conv block 4
    if nb_conv > 3:        
        x = Convolution2D(nb_filters[3], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn4')(x)
        x = ELU()(x)   
        x = MaxPooling2D(pool_size=poolings[3], name='pool4')(x)
        
    # Conv block 5
    if nb_conv == 5:
        x = Convolution2D(nb_filters[4], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn5')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=poolings[4], name='pool5')(x)

    # Flatten the outout of the last Conv Layer
    x = Flatten()(x)
      
    if nb_hidden == 1:
        x = Dropout(dropout)(x)
        x = Dense(dense_units, activation='relu')(x)
    elif nb_hidden == 2:
        x = Dropout(dropout)(x)
        x = Dense(dense_units[0], activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(dense_units[1], activation='relu')(x) 
    else:
        raise ValueError("More than 2 hidden units not supported at the moment.")
    
    # Output Layer
    x = Dense(output_shape, activation=activation, name = 'output')(x)
    
    # Create model
    model = Model(melgram_input, x)
    
    return model


# ### Set model parameters
# 
# **Exercise:** Try to experiment with the following parameters:

# In[85]:


# number of Convolutional Layers (3, 4 or 5)
nb_conv_layers = 3

# number of Filters in each layer (# of elements must correspond to nb_conv_layers)
nb_filters = [32,64,64,128,128]

# number of hidden layers at the end of the model
nb_hidden = 1 # 2

# how many neurons in each hidden layer (# of elements must correspond to nb_hidden)
dense_units = 128 #[128,56]

# how many output units
output_shape = n_classes

# which activation function to use for OUTPUT layer
# IN A SINGLE LABEL MULTI-CLASS TASK with N classes we use softmax activation to BALANCE best between the classes 
# and find the best decision for ONE class
# (in a binary *or* multi-label task we use 'sigmoid')
output_activation = 'softmax'

# which type of normalization
normalization = 'batch'

# how much dropout to use on the hidden dense layers
dropout = 0.2


# In[86]:


model = CompactCNN(input_shape, nb_conv = nb_conv_layers, nb_filters= nb_filters, 
                           normalize=normalization, 
                           nb_hidden = nb_hidden, dense_units = dense_units, 
                           output_shape = output_shape, activation = output_activation, 
                           dropout = dropout)


# In[87]:


input_shape


# In[88]:


model.summary()


# ## Training Parameters
# 
# In contrast with the binary Instrumental vs. Vocal task above we have to do some **important changes**:

# ### Change #1: Loss

# In[89]:


# the loss for a single label classification task is CATEGORICAL crossentropy
loss = 'categorical_crossentropy' 


# ### Change #2: Output activation

# In[90]:



# which activation function to use for OUTPUT layer
# IN A SINGLE LABEL MULTI-CLASS TASK with N classes we use softmax activation to BALANCE best between the classes 
# and find the best decision for ONE class
output_activation = 'softmax'

# Note that this has been set already above in the CompactCNN model definition (changing it here will be impactless)


# ### Optimizer
# 
# We have used **Stochastic Gradient Descent (SGD)** in our first experiments. This is the standard optimizer. A number of advanced algorithms are available.
# 
# **Exercise:** Try various optimizers and their parameters and observe the impact on training convergence.

# In[91]:


# Optimizers

# we define a couple of optimizers here
sgd = optimizers.SGD() # standard
sgd_momentum = optimizers.SGD(momentum=0.9, nesterov=True)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.01)#lr=0.001 decay = 0.03
adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
adam = optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.01)
nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)

# PLEASE CHOOSE ONE:
optimizer = adam


# ### Metrics
# 
# In addition to accuracy, we evaluate precision and recall here.

# In[92]:


# Metrics
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

metrics = ['accuracy', precision, recall]


# ### Other Parameters

# In[93]:


batch_size = 32 

validation_split=0.1 

random_seed = 0

callbacks = None

epochs = 10


# ### Tensorboard (optional)
# 
# Tensorboard (included in Tensorflow) is a web-based visualization to observe your training process.

# In[94]:


from keras.callbacks import TensorBoard


# In[95]:


# set PATH where to store tensorboard files
cwd = os.getcwd()
TB_LOGDIR = join(cwd, "tensorboard")

# make a subdir for each task and another subdir for each run using date/time
from time import strftime, localtime
experiment_name = task #join(task, strftime("%Y-%m-%d_%H-%M-%S", localtime()))


# In[96]:


tb_logdir_cur = os.path.join(TB_LOGDIR, experiment_name)
tb_logdir_cur


# In[97]:


print("Execute the following in a terminal:\n")
print("tensorboard --logdir=" + TB_LOGDIR)


# In[99]:


# initialize TensorBoard in Python
tensorboard = TensorBoard(log_dir = tb_logdir_cur)

# add to Keras callbacks
callbacks = [tensorboard]


# Then open Tensorboard in browser:
# 
# http://localhost:6006

# ## Training

# In[100]:


# Summary of Training options

print(loss)
print(optimizer)
print(metrics)
print("Batch size:", batch_size, "\nEpochs:", epochs)


# In[101]:


# COMPILE MODEL
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)


# In[102]:


# past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
past_epochs = 0


# In[103]:


# START TRAINING

history = model.fit(train_set, train_classes, 
                     validation_split=validation_split,
                     #validation_data=(X_test,y_test), # option to provide separate validation set
                     epochs=epochs, 
                     initial_epoch=past_epochs,
                     batch_size=batch_size, 
                     callbacks=callbacks
                     )

past_epochs += epochs


# ### Testing / Evaluation

# In[104]:


# compute probabilities for the classes (= get outputs of output layer)
test_pred_prob = model.predict(test_set)
test_pred_prob[0:10]


# In[105]:


# for a multi-class SINGLE LABEL OUTPUT classification task, we use ARG MAX to determine 
# the most probable class per instance (we take the ARG MAX of the row vectors)
test_pred = np.argmax(test_pred_prob, axis=1)
test_pred[0:20]


# In[106]:


# do the same for groundtruth
test_gt = np.argmax(test_classes, axis=1)
test_gt[0:20]


# In[107]:


# evaluate Accuracy
accuracy_score(test_gt, test_pred)


# In[108]:


# evaluate Precision
precision_score(test_gt, test_pred, average='micro')


# In[109]:


# evaluate Recall
recall_score(test_gt, test_pred, average='micro')


# In[110]:


print(classification_report(test_gt, test_pred, target_names=metadata.columns))


# # 3) Mood Recognition
# 
# This is a multi-label classification task: multiple categories to detect, any of them can be 0 or 1.

# ## Load Audio Spectrograms
# 
# We prepared already the Mel spectrograms for the audio files used in this task.

# In[111]:


task = 'moods'

# load Mel spectrograms
spectrogram_file = SPECTROGRAM_FILE_PATTERN % task
spectrograms, spectrograms_clip_ids = load_spectrograms(spectrogram_file)

# standardize
data = standardize(spectrograms)
data.shape # verify the shape of the loaded & standardize spectrograms


# ## Load Metadata

# In[112]:


# use META_FILE_PATTERN to load the correct metadata file. set correct METADATA_PATH above
csv_file = LABEL_FILE_PATTERN % task
metadata = pd.read_csv(csv_file, index_col=0) #, sep='\t')
metadata.shape


# In[113]:


metadata.head()


# In[114]:


# how many tracks per mood
metadata.sum()


# In[115]:


# maximum number of moods per track
metadata.sum(axis=1).max()


# ### Align Metadata and Spectrograms

# In[125]:


spec_indices = spectrograms_clip_ids.loc[metadata.index]['spec_id']
data = spectrograms[spec_indices,:]


# ### Create Train X and Y: data and classes

# In[126]:


# classes needs to be a "1-hot encoded" numpy array (which our groundtruth already is! we just convert pandas to numpy)
classes = metadata.values
classes


# In[127]:


n_classes = metadata.shape[1]


# In[128]:


# add channel (see above)
data = add_channel(data)
data.shape


# In[129]:


# input_shape: we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of files)
input_shape = data.shape[1:]  
input_shape


# ### Train & Test Set Split
# 
# We split the original full data set into two parts: Train Set (75%) and Test Set (25%).

# ### Change: We cannot use Stratified Split here as it does not make sense for a Multi-Label task!
# 
# We use a random ShuffleSplit instead.

# In[130]:


# use ShuffleSplit INSTEAD OF StratifiedShuffleSplit 

splitter = ShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes)

for train_index, test_index in splits:
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes[train_index]
    test_classes = classes[test_index]
# Note: this for loop is only executed once if n_splits==1


# In[131]:


print(train_set.shape)
print(test_set.shape)


# ## Model and Training Parameters
# 
# we use the same model as for Instrumental vs. Vocal and Genres above
# 
# with a few changes in the Training parameters

# ### Change #1: Loss

# In[132]:


# the loss for a MULTI label classification task is BINARY crossentropy
loss = 'binary_crossentropy' 


# ### Change #2: Output activation

# In[133]:


# which activation function to use for OUTPUT layer
# IN A MULTI-LABEL TASK with N classes we use SIGMOID activation same as with a BINARY task
# as EACH of the classes can be 0 or 1 

output_activation = 'sigmoid'


# ## Model
# 
# We are reusing the **CompactCNN** from above.
# 
# **Exercise:** Adapt the parameters of the CompactCNN model:

# In[134]:


# number of Convolutional Layers (3, 4 or 5)
nb_conv_layers = 3

# number of Filters in each layer (# of elements must correspond to nb_conv_layers)
nb_filters = [32,64,64,128,128]

# number of hidden layers at the end of the model
nb_hidden = 1 # 2

# how many neurons in each hidden layer (# of elements must correspond to nb_hidden)
dense_units = 128 #[128,56]

# how many output units
output_shape = n_classes

# which type of normalization
normalization = 'batch'

# how much dropout to use on the hidden dense layers
dropout = 0.2


# In[135]:


model = CompactCNN(input_shape, nb_conv = nb_conv_layers, nb_filters= nb_filters, 
                           normalize=normalization, 
                           nb_hidden = nb_hidden, dense_units = dense_units, 
                           output_shape = output_shape, activation = output_activation, 
                           dropout = dropout)


# In[136]:


model.summary()


# ### TensorBoard setup (optional)

# In[137]:


experiment_name = task

tb_logdir_cur = os.path.join(TB_LOGDIR, experiment_name)

# initialize TensorBoard in Python
tensorboard = TensorBoard(log_dir = tb_logdir_cur)

# + add to callbacks
callbacks = [tensorboard]

# otherwise assign:
# callbacks = None


# ### Rest of Parameters
# 
# stay essentially the same (or similar)
# 
# **Excercise:** change the optimizer (see same exercise in Genre model)

# In[138]:


# Optimizer
optimizer = adam

metrics = ['accuracy']

random_seed = 0

batch_size = 32 

validation_split = 0.1 

epochs = 10


# ## Training

# In[139]:


# Summary of Training options

print(loss)
print(optimizer)
print(metrics)
print("Batch size:", batch_size, "\nEpochs:", epochs)


# In[140]:


# COMPILE MODEL
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)


# In[141]:


# past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
past_epochs = 0


# In[142]:


# START TRAINING

history = model.fit(train_set, train_classes, 
                     validation_split=validation_split,
                     #validation_data=(X_test,y_test), 
                     epochs=epochs, 
                     initial_epoch=past_epochs,
                     batch_size=batch_size, 
                     callbacks=callbacks
                     )

past_epochs += epochs


# ### Evaluation on Test Set

# In[143]:


# compute probabilities for the classes (= get outputs of output layer)
test_pred_prob = model.predict(test_set)
test_pred_prob[0:10]


# #### Change: In a multi-label task we have to round each prediction probability to 0 or 1

# In[144]:


# to get the predicted class(es) we have to round 0 < 0.5 > 1
test_pred = np.round(test_pred_prob)
test_pred[0:10]


# In[145]:


# groundtruth
test_classes[0:10]


# ### Evaluation Metrics
# 
# In addition to Accuracy, common metrics for multi-label classification are ROC AUC score and Hamming Loss (among others).

# In[146]:


# Accuracy
accuracy_score(test_classes, test_pred)


# In[147]:


# Area Under the Receiver Operating Characteristic Curve (ROC AUC) 
roc_auc_score(test_classes, test_pred)


# In[148]:


# Hamming loss is the fraction of labels that are incorrectly predicted.
hamming_loss(test_classes, test_pred)

