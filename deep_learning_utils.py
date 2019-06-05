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
	for song_id in song_inds[30:]:
		print(song_id, JAAH_INFO.audio_exists.loc[song_id])
		if JAAH_INFO.audio_exists.loc[song_id]:
			compute_feat_for_file(song_id, 'mfsg')


def standardize(data):
	# vectorize before standardization (cause scaler can't do it in that format)
	N, ydim, xdim = data.shape
	data = data.reshape(N, xdim*ydim)
	# standardize
	scaler = sklearn.preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	# reshape to original shape
	return data.reshape(N, ydim, xdim)

# DONE: normalizing time windows doesn't make sense to me because there could be arbitrary time shifts.
# Scaling to frequency bins does make some sense though.
# So maybe collect all the bins for a given frequency (across the dataset) and normalize that as a single attribute?
def standardize_fbins_only(data):
	# vectorize before standardization (cause scaler can't do it in that format)
	# data = np.reshape(np.arange(30),[2,3,5])
	N, ydim, xdim = data.shape
	data = data.transpose((0,2,1)).reshape(N*xdim, ydim)
	# standardize
	scaler = sklearn.preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	# reshape to original shape
	return data.reshape(N, xdim, ydim).transpose((0,2,1))

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

def collect_info(song_id, layer='funcs'):
	global INFO_DF
	assert layer in INFO_DF.columns
	MFSG_SONG_PATH_PATTERN = os.path.join(MY_BASE_PATH, 'jaah_song_%s_mfsg.npz')
	data = np.load(MFSG_SONG_PATH_PATTERN % song_id)['feat']
	times = np.load(MFSG_SONG_PATH_PATTERN % song_id)['t']
	song_id_vector = np.ones(len(times))*song_id
	onehot_columns, meta_categories = make_1hot_for_col([song_id], layer, times)
	return data, times, song_id_vector, onehot_columns, meta_categories

def subsample_lists(array_list, n_to_keep=None, percent_keep=None, shuffle=True):
	assert ((n_to_keep is not None) or (percent_keep is not None))
	indices = range(array_list[0].shape[0])
	if shuffle:
		indices = sp.random.permutation(indices)
	if n_to_keep is not None:
		indices = indices[:n_to_keep]
	elif percent_keep is not None:
		n_to_keep = int(len(indices)*percent_keep)
		indices = indices[:n_to_keep]
	return [item[indices] for item in array_list]

def script_2_load_percent_of_all_info(song_id_list, layer, percent_keep):
	all_lists = [subsample_lists(collect_info(song_id,layer)[:-1], percent_keep=percent_keep) for song_id in song_id_list]
	images, times, songvecs, onehots = zip(*all_lists)
	images = np.concatenate(images)
	images_proc = add_channel(standardize_fbins_only(images))
	songvecs = np.concatenate(songvecs)
	onehots = np.concatenate(onehots)
	metadata_columns = collect_info(song_id_list[0],layer)[-1]
	return images_proc, songvecs, onehots, metadata_columns

def split_data_with_indices(input_data, class_data, block_data, test_indices):
	test_index = np.zeros_like(block_data).astype(int)
	for ti in test_indices:
		test_index[block_data==ti] = True
	train_index = 1-test_index
	train_X = input_data[train_index==1, :, :, :]
	test_X = input_data[test_index==1, :, :, :]
	train_y = class_data[train_index==1,:]
	test_y = class_data[test_index==1,:]
	return train_X, test_X, train_y, test_y

def script_for_basic_model(input_shape, n_layers=2):
	model = keras.models.Sequential()
	conv_filters = 32   # number of convolution filters (= CNN depth)
	# 1st Layer
	for _ in range(n_layers):
		model.add(keras.layers.Convolution2D(conv_filters, (3, 3), input_shape=input_shape))
		model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
	# Dropout
	model.add(keras.layers.Dropout(rate=0.25))
	# rate = (1 - keep_prob)
	model.add(keras.layers.Flatten()) 
	model.add(keras.layers.Dense(256, activation='sigmoid')) 
	model.add(keras.layers.Dense(n_classes,activation='sigmoid'))
	return model


def CompactCNN(input_shape, nb_conv, nb_filters, normalize, nb_hidden, dense_units, 
			   output_shape, activation, dropout, multiple_segments=False, input_tensor=None):
	from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
	from keras.layers.normalization import BatchNormalization
	from keras.layers.advanced_activations import ELU
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
	# Conv blocks
	for i in range(nb_conv):
		x = Convolution2D(nb_filters[nb_conv], (3, 3), padding='same')(x)
		x = BatchNormalization(axis=channel_axis, name='bn'+str(i+1))(x)
		x = ELU()(x)
		x = MaxPooling2D(pool_size=poolings[i], name='pool'+str(i+1))(x)
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
	model = keras.models.Model(melgram_input, x)
	return model
