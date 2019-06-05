from deep_learning_utils import *

# Next up:
# - compute feats over a bunch of files
# - concatenate feats and categories over files
# - normalize features (create normalization function and apply)
# - create test/train splits over files
# - test algorithm on files


MFSG_SONG_PATH_PATTERN = os.path.join(MY_BASE_PATH, 'jaah_song_%s_mfsg.npz')
audio_exists = JAAH_INFO.loc[JAAH_INFO.audio_exists].iloc[:,0].values
song_id_list = sorted(list(set.intersection(set(audio_exists), set(np.unique(INFO_DF.ind)))))
images_proc, songvecs, onehots, metadata_columns = script_2_load_percent_of_all_info(song_id_list, 'instruments', percent_keep=0.1)

# Add a 'no instrument' column
no_instrument = np.all(1-onehots,axis=1)*1
onehots_full = np.insert(onehots,onehots.shape[1],no_instrument,axis=1)
metadata_columns += ['no_instrument']

brass = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["trumpet", "trombone", "horn", "cornet"]]],axis=1)
reed = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["clarinet", "sax"]]],axis=1)
pluck = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["banjo", "guitar", "bass"]]],axis=1)
percussion = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["drums", "sax"]]],axis=1)
keyboard = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["piano", "vibraphone"]]],axis=1)
vocal = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["vocals", "scat"]]],axis=1)
other = np.sum(onehots[:,[metadata_columns.index(instr) for instr in ["ensemble"]]],axis=1)
metadata = np.stack((brass, reed, pluck, percussion, keyboard, vocal, other),axis=1)
# metadata = metadata.append(1-np.max(metadata,axis=1))
# ['banjo', 'bass', 'clarinet', 'cornet', 'drums', 'ensemble', 'guitar', 'horn', 'piano', 'sax', 'scat', 'trombone', 'trumpet', 'vibraphone', 'vocals', 'no_instrument']
metadata = np.stack((brass, 1-brass),axis=1)
# Define solo vs. non-solo detection task:
# solo = 1*(1<= (metadata_full.solo + metadata_full.improvisation))
# metadata = np.stack((solo,1-solo),axis=1)
input_shape = images_proc.shape[1:]
train_ranges, test_ranges, n_splits = produce_splits(song_id_list, 0.05)

results = []
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems
optimizer = 'sgd' 
metrics = ['accuracy']
batch_size = 32
epochs = 5
for split_i in range(5):
	train_X, test_X, train_y, test_y = split_data_with_indices(images_proc, metadata, songvecs, test_ranges[split_i])
	n_classes = metadata.shape[1]
	model = script_for_basic_model(input_shape, n_layers=2)
	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
	history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)
	test_pred = model.predict_classes(test_X)
	# test_pred[0:10]
	pred_split = np.sum(test_pred)/len(test_pred)
	# 1 layer
	acc = sklearn.metrics.accuracy_score(test_y[:,1], test_pred)
	# Baseline performance:
	baseline_acc = np.max(np.sum(metadata,axis=0))/np.sum(metadata)
	results += [[acc, baseline_acc, pred_split, history, model]]
	print(results)

# Example results after 5 epochs:
# >>> results
# [[0.8663003663003663, 0.8397278397278397, 0.9642857142857143, <keras.callbacks.History object at 0x130e63b00>, <keras.engine.sequential.Sequential object at 0x134555198>]]
# >>> history.history
# {'loss': [0.4264688369071449, 0.38982690172228984, 0.3808055007847274, 0.3722048993402934, 0.3642222402919396], 'acc': [0.8266693728193202, 0.8429064339798154, 0.8476422434004449, 0.8528516338986301, 0.8575197889706285]}





##### All-instrument classification

# Loading data, same as before
MFSG_SONG_PATH_PATTERN = os.path.join(MY_BASE_PATH, 'jaah_song_%s_mfsg.npz')
audio_exists = JAAH_INFO.loc[JAAH_INFO.audio_exists].iloc[:,0].values
song_id_list = sorted(list(set.intersection(set(audio_exists), set(np.unique(INFO_DF.ind)))))
images_proc, songvecs, onehots, metadata_columns = script_2_load_percent_of_all_info(song_id_list, 'instruments', percent_keep=0.05)
# Add a 'no instrument' column
no_instrument = np.all(1-onehots,axis=1)*1
onehots_full = np.insert(onehots,onehots.shape[1],no_instrument,axis=1)
metadata_columns += ['no_instrument']

# Train/test splits
input_shape = images_proc.shape[1:]
train_ranges, test_ranges, n_splits = produce_splits(song_id_list, 0.05)
classes = metadata
n_classes = classes.shape[1]

results = []
loss = 'binary_crossentropy' 
output_activation = 'sigmoid'
optimizer = 'sgd' 
metrics = ['accuracy']
batch_size = 32
epochs = 5
validation_split = 0.1
for split_i in range(5):
	train_X, test_X, train_y, test_y = split_data_with_indices(images_proc, metadata, songvecs, test_ranges[split_i])
	n_classes = metadata.shape[1]
	model = script_for_basic_model(input_shape, n_layers=2)
	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
	history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
	test_pred = model.predict_classes(test_X)
	# test_pred[0:10]
	pred_split = np.sum(test_pred)/len(test_pred)
	# 1 layer
	acc = sklearn.metrics.accuracy_score(test_y[:,1], test_pred)
	# Baseline performance:
	baseline_acc = np.max(np.sum(metadata,axis=0))/np.sum(metadata)
	results += [[acc, baseline_acc, pred_split, history, model]]
	print(results)

# number of Convolutional Layers (3, 4 or 5)
nb_conv_layers = 3
# number of Filters in each layer (# of elements must correspond to nb_conv_layers)
nb_filters = [32,64,64,128,128]
# number of hidden layers at the end of the model
nb_hidden = 1 # 2
# how many neurons in each hidden layer (# of elements must correspond to nb_hidden)
dense_units = 128 #[128,56]
output_shape = n_classes
normalization = 'batch'
# how much dropout to use on the hidden dense layers
dropout = 0.2
model = CompactCNN(input_shape, nb_conv = nb_conv_layers, nb_filters= nb_filters, normalize=normalization, nb_hidden = nb_hidden, dense_units = dense_units, output_shape = output_shape, activation = output_activation, dropout = dropout)
model.summary()

# experiment_name = "all_instrument_solo_classification"
# tb_logdir_cur = os.path.join(TB_LOGDIR, experiment_name)
# # initialize TensorBoard in Python
# tensorboard = TensorBoard(log_dir = tb_logdir_cur)
# # + add to callbacks
# callbacks = [tensorboard]
#
# otherwise assign:
callbacks = None
# Optimizer
adam = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.01)
optimizer = adam
metrics = ['accuracy']
random_seed = 0
batch_size = 32 
validation_split = 0.1 
epochs = 10

print(loss)
print(optimizer)
print(metrics)
print("Batch size:", batch_size, "\nEpochs:", epochs)

model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

train_X, test_X, train_y, test_y = split_data_with_indices(images_proc, metadata, songvecs, test_ranges[split_i])

# past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
past_epochs = 0
history = model.fit(train_X, train_y, 
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit





# # 2) Genre Classification
# 
# In this Genre classification task, we have multiple classes, but the decision has to be made for 1 target class.
# This is called a single-label / multi-class task (as opposed to a multi-label task).

# Stratified Split retains the class balance in both sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes)

for train_index, test_index in splits:
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes[train_index]
    test_classes = classes[test_index]
# Note: this for loop is only executed once if n_splits==1


# ## Model: Compact CNN
# 
# This is a 5 layer Convolutional Neural Network inspired and adapted from Keunwoo Choi (https://github.com/keunwoochoi/music-auto_tagging-keras)
# 
# * It is specified using Keras' functional Model **Graph API** (https://keras.io/models/model/).
# * It allows to specify 3, 4 or 5 Convolutional Layers.
# * It adapts the Pooling sizes according to the number of Mel bands use in the input.
# * It uses Batch Normalization.

# In[84]:



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

