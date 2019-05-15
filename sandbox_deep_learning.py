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
import importlib
# sys.path.append(os.path.expanduser("~/Dropbox/Ircam/code"))
import segmenter
import jaah_experiment as jaah
import sklearn
import keras

JAAH_INFO = pd.read_csv("audio_paths_index.csv")
INFO_DF = jaah.parse_all_annotations()
MY_BASE_PATH = os.path.expanduser("~/Documents/data/JAAH/feature_data/")

def load_audio_for_song_id(song_id):
	global JAAH_INFO
	assert JAAH_INFO.audio_exists.loc[song_id]
	audio_path = JAAH_INFO.audio_path.loc[song_id]
	y, sr = librosa.core.load(audio_path, sr=22050)
	return y, sr

def arrange_feat_images(feat, window_size_secs=3):
	# To take 5 second windows of MFCC feature frames, we need 5 seconds / (0.0929/4 sec/hop) = 216 frames
	feat_times = librosa.core.frames_to_time(range(feat.shape[1]), n_fft=2048)
	# Note the hard-coded parameters, which are the librosa defaults:
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
	feat_image_times = np.sort(feat_image_times)
	return feat_images, feat_image_times

def make_1hot_for_col(song_id, col, times):
	global INFO_DF
	assert col in INFO_DF.columns
	vocab_words = sorted(np.unique([i for j in INFO_DF[col] for i in j]))
	onehot = np.zeros((len(times),len(vocab_words)))
	info_rows = INFO_DF.loc[INFO_DF.ind==song_id]
	for i in range(info_rows.shape[0]):
		on,off,row_words = info_rows.iloc[i][['onset','offset',col]]
		rel_range = (on <= times) & (times < off)
		onehot_vector = np.array([word in row_words for word in vocab_words])
		onehot[rel_range] = onehot_vector
	return onehot, vocab_words

def compute_feat_for_file(song_id, feat_name='mfsg'):
	global MY_BASE_PATH
	SONG_PATH_PATTERN = os.path.join(MY_BASE_PATH, 'jaah_song_%s_%s.npz')
	y, sr = load_audio_for_song_id(song_id)
	assert feat_name in ['mfsg','cqt']
	if feat_name == 'mfsg':
		feat = librosa.feature.melspectrogram(y, n_fft=2048, hop_length=512)  # 0.0929 second windows with 1/4 window overlap
	elif feat_name == 'cqt':
		feat = librosa.core.cqt(y, n_fft=2048, hop_length=512)
	images, times = arrange_feat_images(feat,3)
	images = np.transpose(images,[1,0,2])
	np.savez_compressed(SONG_PATH_PATTERN % (song_id, feat_name), feat=images, t=times)

def script_1_computing_mfsg():
	global INFO_DF
	song_inds = np.unique(INFO_DF.ind)
	for song_id in song_inds[:30]:
		print(song_id, JAAH_INFO.audio_exists.loc[song_id])
		if JAAH_INFO.audio_exists.loc[song_id]:
			compute_feat_for_file(song_id, 'mfsg')


# TODO: normalizing time windows doesn't make sense to me because there could be arbitrary time shifts.
# Scaling to frequency bins does make some sense though.
# So maybe collect all the bins for a given frequency (across the dataset) and normalize that as a single attribute?
def standardize(data):
	# vectorize before standardization (cause scaler can't do it in that format)
	N, ydim, xdim = data.shape
	data = data.reshape(N, xdim*ydim)
	# standardize
	scaler = sklearn.preprocessing.StandardScaler()
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

def produce_splits(id_list, test_percent):
	tmp_list = sp.random.permutation(id_list)
	n_test_ids = np.ceil(len(tmp_list) * test_percent).astype(int)
	first_is = np.arange(0,len(id_list),n_test_ids)
	last_is = np.append(first_is[1:],len(tmp_list))
	test_ranges = [tmp_list[first_is[i]:last_is[i]] for i,_ in enumerate(first_is)]
	train_ranges = [[i for i in tmp_list if i not in test_range] for test_range in test_ranges]
	n_splits = len(first_is)
	return train_ranges, test_ranges, n_splits


# Next up:
# - compute feats over a bunch of files
# - concatenate feats and categories over files
# - normalize features (create normalization function and apply)
- create test/train splits over files
- test algorithm on files


MFSG_SONG_PATH_PATTERN = os.path.join(MY_BASE_PATH, 'jaah_song_%s_mfsg.npz')
audio_exists = JAAH_INFO.loc[JAAH_INFO.audio_exists].iloc[:,0].values
song_id_list = sorted(list(set.intersection(set(audio_exists), set(np.unique(INFO_DF.ind)))))
song_id_list = song_id_list[:25]
all_data = np.concatenate([np.load(MFSG_SONG_PATH_PATTERN % song_id)['feat'] for song_id in song_id_list], axis=0)
all_data_proc = add_channel(standardize(all_data))
all_times_list = [np.load(MFSG_SONG_PATH_PATTERN % song_id)['t'] for song_id in song_id_list]
all_meta_songid = np.concatenate([np.ones((len(all_times_list[i])))*song_id_list[i] for i in range(len(all_times_list))])
all_meta_onehots = np.concatenate([make_1hot_for_col(song_id_list[i], 'funcs', all_times_list[i])[0] for i in range(len(all_times_list))], axis=0)
onehot_columns = make_1hot_for_col(song_id_list[0], 'funcs', all_times_list[0])[1]
metadata_full = pd.DataFrame(all_meta_onehots, columns=onehot_columns)
# Define solo vs. non-solo detection task:
solo = 1*(1<= (metadata_full.solo + metadata_full.improvisation))
nonsolo = 1-solo
metadata = np.stack((solo,nonsolo),axis=1)
input_shape = all_data_proc.shape[1:]
train_ranges, test_ranges, n_splits = produce_splits(song_id_list, 0.25)

split_i = 0
test_index = np.zeros_like(all_meta_songid).astype(int)
for ti in test_ranges[split_i]:
	test_index[all_meta_songid==ti] = True

train_index = 1-test_index
train_X = all_data_proc[train_index==1, :, :, :]
test_X = all_data_proc[test_index==1, :, :, :]
train_y = metadata[train_index==1,:]
test_y = metadata[test_index==1,:]
n_classes = metadata.shape[1]

#np.random.seed(0) # make results repeatable

model = keras.models.Sequential()
conv_filters = 32   # number of convolution filters (= CNN depth)
# 1st Layer
model.add(keras.layers.Convolution2D(conv_filters, (3, 3), input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
# 2nd Layer
model.add(keras.layers.Convolution2D(conv_filters, (3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
model.add(keras.layers.Convolution2D(conv_filters, (3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
# Dropout
model.add(keras.layers.Dropout(rate=0.25))
# rate = (1 - keep_prob)
model.add(keras.layers.Flatten()) 
model.add(keras.layers.Dense(256, activation='sigmoid')) 
model.add(keras.layers.Dense(n_classes,activation='sigmoid'))

model.summary()

loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems
optimizer = 'sgd' 
metrics = ['accuracy']
batch_size = 32
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

epochs = 1
history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)

test_pred = model.predict_classes(test_X)
test_pred[0:10]
np.sum(test_pred)/len(test_pred)
# 1 layer
sklearn.metrics.accuracy_score(test_y[:,0], test_pred)


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
