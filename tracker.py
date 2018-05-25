# Beat and downbeat detection sandbox

#
# 
# 	Imports

import dataset
import essentia
import essentia.standard
import essentia.streaming
import glob
import json
import Levenshtein
import librosa
import madmom
import mir_eval
import numpy as np
import os.path
import pandas as pd
# import pickle
# import py_sonicvisualiser
import re
import scipy as sp
import vamp


class RhythmData(object):
	def __init__(self, beat_onset, bar=[], beat=[], infer=True):
		assert (len(beat_onset)>0), "You must provide onsets to instantiate a RhythmData object"
		self.beat_onset = np.array(beat_onset).astype(float)
		self.bar = np.array(bar).astype(int)
		self.beat = np.array(beat).astype(int)
		# self.tempo = tempo.astype(float)
		# self.time_signature = time_signature
		if infer:
			self.infer_missing_data()
		
	def infer_missing_data(self):
		# If we have nothing at all to go on, just assume it's 4/4, with the first beat = 1.
		if not len(self.bar) and not len(self.beat):
			time_sig = 4
			self.beat = [range(time_sig)[np.mod(i,time_sig)]+1 for i in range(len(self.beat_onset))]
		
		# If we have beat indices and beat_onset, we can infer the downbeat onsets and bar numbers.
		if len(self.beat) and not len(self.bar):
			# bars = np.cumsum(np.array(new_rhythm_data.beat[1:])<np.array(new_rhythm_data.beat[:-1]))
			bars = np.cumsum(np.array(self.beat[1:])<np.array(self.beat[:-1]))
			bars = np.concatenate(([0], bars))
			self.bar = bars
		
		# If we have the bar number of each beat, we can infer the beat indices.
		if len(self.bar) and not len(self.beat):
			beats = np.arange(1,len(self.bar)+1)
			for val in np.unique(self.bar):
				beats[self.bar==val] -= np.min(beats[self.bar==val])-1
			self.beat = beats
	
	def as_dataframe(self):
		return pd.DataFrame({'bar':self.bar, 'beat':self.beat, 'onset':self.beat_onset})

class Beat(object):

	# Initial management:
	# - Setting up paths
	# - Loading beats
	# - Loading audio

	def __init__(self):
		self.data_dir = "/Users/jordan/Documents/data/WeimarJazzDatabase"
		self.database_path = self.data_dir + "/wjazzd.db"
		self.output_database_path = self.data_dir + "/output_database.db"
		self.sv_dir = self.data_dir + "/annotations/SV/"
		self.annotation_dir = self.data_dir + "/annotations/"
		self.audio_orig_dir = self.data_dir + "/audio/wav_orig/"
		self.audio_solo_dir = self.data_dir + "/audio/wav_solo/"
		self.beats_dir = self.data_dir + "/annotations/beats/"
		self.solo_dir = self.data_dir + "/annotations/solo/"
		self.est_dir = self.data_dir + "/estimates/"
		# self.manage_metadata()
		self.load_db()
		# Open metadata and extract contents
		self.sv_paths = glob.glob(self.sv_dir + "*.sv")
		song_basenames = [re.sub("_FINAL.sv","",re.sub(self.sv_dir,"",path)) for path in self.sv_paths]
		self.stripped_basenames = song_basenames[:]
		for i in range(len(song_basenames)):
		    if song_basenames[i][-2] == "-":
		        self.stripped_basenames[i] = song_basenames[i][:-2]
		self.fs = 44100
		self.n_fft = 2048
		self.hop_length = 512
		self.raw_data = {}		# All raw algorithm outputs will be filed under here by name.
		self.beats = {}			# All entries here will be RhythmData objects, probably converted from a corresponding raw data entry.
	
	def set_index(self, ind):
		self.load_audio(ind)
		self.load_true_beats_and_downbeats(ind)
		self.ind = ind
	
	def make_beat_path(self, beat_type = 0, extractor_type = 'madmom'):
		if beat_type == 0:
			beat_string = 'beats'
		elif beat_type == 1:
			beat_string = 'downbeats'
		return self.est_dir + extractor_type + "/" + str(self.ind) + "-" + beat_string + ".txt"
	
	def make_csv_path(self, extractor_type = 'madmom'):
		return self.est_dir + extractor_type + "/" + str(self.ind) + ".csv"
	
	# def make_pickle_path(self, extractor_type = "madmom"):
	# 	return self.est_dir + extractor_type + "/" + str(self.ind) + ".p"

	# def manage_metadata(self):
	# 	metadata_table_path = self.data_dir + "/annotations/weimar_contents.tsv"
	# 	metadata_table = open(metadata_table_path,'r').readlines()
	# 	header = metadata_table[0].strip().split("\t")
	# 	datarows = [line.strip().split("\t") for line in metadata_table[1:]]
	# 	data = pd.DataFrame(datarows, columns=header)
	# 	data['Year'] = data['Year'].astype(int)
	# 	data['Tempo'] = data['Tempo'].astype(float)
	# 	data['Tones'] = data['Tones'].astype(int)
	# 	data['orig_path'] = None
	# 	audio_filepaths = glob.glob(self.audio_solo_dir + "*.wav")
	# 	audio_filenames = [os.path.basename(filename) for filename in audio_filepaths]
	# 	best_inds = np.zeros(len(data.index)).astype(int)
	# 	for i in data.index:
	# 		predicted_path = re.sub(" ","",data['Performer'][i]) + "_" + re.sub(" ","",data['Title'][i]) + "_Orig.wav"
	# 		edit_distances = [Levenshtein.distance(predicted_path,af) for af in audio_filenames]
	# 		# if np.min(edit_distances)<10:
	# 		best_inds[i] = int(np.argmin(edit_distances))
	# 	data['orig_path'] = [audio_filenames[i] for i in best_inds]
	# 	data.index = [int(i) for i in data['Ind']]
	# 	self.data = data

	def load_db(self):
		self.db = dataset.connect('sqlite:///' + self.database_path)
	
	# def load_beats(self, ind):
	# 	print "Loading beats."
	# 	beat_file_path = self.beats_dir + str(ind) + ".csv"
	# 	rhythm_data = self.read_csv_format(beat_file_path)
	# 	self.raw_data['true'] = rhythm_data
	
	# def load_true_beats_and_downbeats(self, ind):
	# 	print "Loading beats and downbeats."
	# 	beat_file_path = self.annotation_dir + "magdalena/beat/" + str(ind) + ".txt"
	# 	downbeat_file_path = self.annotation_dir + "magdalena/downbeat/" + str(ind) + ".txt"
	# 	beat_list = list(pd.read_csv(beat_file_path,header=-1)[0])
	# 	downbeat_list = list(pd.read_csv(downbeat_file_path,header=-1)[0])
	# 	downbeat_inds = [i for i in range(len(beat_list)) if np.min(np.abs(beat_list[i]-np.array(downbeat_list)))<0.001]
	# 	beat_labels = np.zeros(len(beat_list)).astype(int)
	# 	beat_labels[downbeat_inds]=1
	# 	for i in range(1,len(beat_labels)-1):
	# 		if beat_labels[i]==0:
	# 			beat_labels[i]=beat_labels[i-1]+1
	#
	# 	new_rhythm_data = pd.DataFrame(columns=['bar','beat','onset'])
	# 	new_rhythm_data.onset = beat_list
	# 	new_rhythm_data.beat = beat_labels
	# 	bars = np.cumsum(np.array(new_rhythm_data.beat[1:])<np.array(new_rhythm_data.beat[:-1]))
	# 	new_rhythm_data.bar[0] = 0
	# 	new_rhythm_data.bar[1:] = bars
	# 	self.rhythm_data['true'] = new_rhythm_data
	
	def load_true_beats_and_downbeats(self, ind):
		relevant_beats = self.db['beats'].find(melid=ind)
		colnames = relevant_beats.keys
		data_list_of_lists = [row for row in relevant_beats]
		table_as_df = pd.DataFrame(data=data_list_of_lists, columns=colnames)
		self.beats['true'] = RhythmData(beat_onset = table_as_df.onset, beat=table_as_df.beat, bar=table_as_df.bar)
		# self.beats['true'] = table_as_df[['bar','beat','onset']]

	def load_audio(self, ind=None, abspath=None):
		print "Loading audio..."
		if abspath is not None:
			self.signal, self.fs = librosa.core.load(abspath, sr=self.fs, mono=False)
			print "Loaded audio."
		elif ind is not None:
			db_entry = self.db['transcription_info'].find_one(melid=ind)
			if db_entry is not None:
				audiopath = self.audio_solo_dir + db_entry['filename_solo'] + ".wav"
				self.signal, self.fs = librosa.core.load(audiopath, sr=self.fs, mono=False)
			else:
				print "Sorry, we could not find matching audio for that file index."
				return
		self.signal_mono = librosa.to_mono(self.signal)
		self.S_mono = librosa.core.stft(self.signal_mono, n_fft=self.n_fft, hop_length=self.hop_length)
		self.V_mono = np.abs(self.S_mono)
		print "Audio loaded and spectrum precomputed."
		
	def estimate_beats(self, extractor_type='qm'):
		assert extractor_type in ['qm','madmom','essentia']
		if extractor_type == 'qm':
			print "Extracting beats using QM Vamp plugin..."
			self.raw_data[extractor_type] = vamp.collect(self.signal_mono, self.fs, 'qm-vamp-plugins:qm-barbeattracker')    # Beat and downbeat
		elif extractor_type == 'essentia':
			print "Extracting beats using Essentia..."
			beat_tracker = essentia.standard.BeatTrackerMultiFeature()
			# ticks, confidence = beat_tracker(audio_ess)
			ticks, confidence = beat_tracker(self.signal_mono)
			self.raw_data[extractor_type] = ticks
		elif extractor_type == 'madmom':
			print "Extracting beats using Madmom..."
			mm_db_detect_func = madmom.features.beats.RNNDownBeatProcessor()(self.signal_mono)
			self.raw_data[extractor_type] = madmom.features.beats.DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)(mm_db_detect_func)
		else:
			print "Unrecognized beat tracker type. Didn't do anything."
			return
		self.set_rhythm(extractor_type)

	def set_rhythm(self, extractor_type='qm'):
		assert extractor_type in ['qm','madmom','essentia']
		self.beat_tracker = extractor_type
		if extractor_type=='qm':
			times,labels = zip(*[(float(item['timestamp']), int(item['label'])) for item in self.raw_data['qm']['list']])
			self.beats[extractor_type] = RhythmData(beat = labels, beat_onset = times, infer=True)
		elif extractor_type=='madmom':
			self.beats[extractor_type] = RhythmData(beat=self.raw_data['madmom'][:,1], beat_onset=self.raw_data['madmom'][:,0], infer=True)
		elif extractor_type=='essentia':
			self.beats[extractor_type] = RhythmData(beat_onset = self.raw_data['essentia'], infer=True)

	# def set_rhythm_from_madmom(self):
	# 	new_rhythm_data = pd.DataFrame(columns=['bar','beat','onset'])
	# 	new_rhythm_data.beat = self.mm_output[:,1].astype(int)
	# 	new_rhythm_data.onset = self.mm_output[:,0]
	# 	bars = np.cumsum(np.array(new_rhythm_data.beat[1:])<np.array(new_rhythm_data.beat[:-1]))
	# 	new_rhythm_data.bar[0] = 0
	# 	new_rhythm_data.bar[1:] = bars
	# 	self.rhythm_data['madmom'] = new_rhythm_data
	#
	# def set_rhythm_from_qm(self):
	# 	# qm_output = vamp.collect(self.signal_mono, self.fs, 'qm-vamp-plugins:qm-barbeattracker')    # Beat and downbeat
	# 	# bt2 = vamp.collect(signal_mono, sr_lib, 'beatroot-vamp:beatroot')               # Beat
	# 	# bt3 = vamp.collect(signal_mono, sr_lib, 'qm-vamp-plugins:qm-tempotracker')      # Beat and tempo
	# 	times,labels = zip(*[(float(item['timestamp']), int(item['label'])) for item in self.qm_output['list']])
	# 	new_rhythm_data = pd.DataFrame(columns=['bar','beat','onset'])
	# 	new_rhythm_data.beat = labels
	# 	new_rhythm_data.onset = times
	# 	bars = np.cumsum(np.array(new_rhythm_data.beat[1:])<np.array(new_rhythm_data.beat[:-1]))
	# 	new_rhythm_data.bar[0] = 0
	# 	new_rhythm_data.bar[1:] = bars
	# 	self.rhythm_data['qm'] = new_rhythm_data
	#
	# def set_rhythm_from_essentia(self):
	# 	new_rhythm_data = pd.DataFrame(columns=['bar','beat','onset'])
	# 	new_rhythm_data.onset = self.es_output
	# 	# Infer downbeat labels naively
	# 	beat = [[1,2,3,4][np.mod(i,4)] for i in range(len(new_rhythm_data.onset))]
	# 	new_rhythm_data.beat = beat
	# 	bars = np.cumsum(np.array(new_rhythm_data.beat[1:])<np.array(new_rhythm_data.beat[:-1]))
	# 	new_rhythm_data.bar[0] = 0
	# 	new_rhythm_data.bar[1:] = bars
	# 	self.rhythm_data['essentia'] = new_rhythm_data
	
	# def set_rhythm_from_madmom(self):
	# 	self.rhythm_data['madmom'] = RhythmData(beat=self.mm_output[:,1], beat_onset=self.mm_output[:,0], infer=True)
	# 	# This will infer the bar numbers [0, 1, 2, ..., nbars] automatically from the beat indices [1,2,3,4,1,2,3,4,1,2,...].
	#
	# def set_rhythm_from_qm(self):
	# 	times,labels = zip(*[(float(item['timestamp']), int(item['label'])) for item in self.qm_output['list']])
	# 	self.rhythm_data['qm'] = RhythmData(beat = labels, beat_onset = times, infer=True)
	#
	# def set_rhythm_from_essentia(self):
	# 	self.rhythm_data['essentia'] = RhythmData(beat_onset = self.es_output, infer=True)
	
	def read_csv_format(self, beat_file_path):
		csv_data = pd.read_csv(beat_file_path,header=0)
		csv_data[['bar','beat']] = csv_data[['bar','beat']].astype(int)
		csv_data['onset'] = csv_data['onset'].astype(float)
		return csv_data
	
	def write_csv_format(self, extractor):
		write_path = self.make_csv_path(extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			self.beats[extractor].as_dataframe().to_csv(filehandle, index=False)
	
	def write_beats(self, extractor):
		write_path = self.make_beat_path(beat_type = 0, extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			filehandle.write("\n".join(list(self.beats[extractor].beat_onset.astype(str))))
	
	def write_downbeats(self, extractor):
		write_path = self.make_beat_path(beat_type = 1, extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			# self.rhythm_data[extractor].to_csv(filehandle)
			filehandle.write("\n".join(list(self.beats[extractor].beat_onset[self.beats[extractor].beat==1].astype(str))))
			
	def run_estimates(self):
		self.estimate_beats(extractor_type='qm')
		self.estimate_beats(extractor_type='essentia')
		self.estimate_beats(extractor_type='madmom')
	
	def write_all_rhythms(self):
		self.write_beats('qm')
		self.write_beats('essentia')
		self.write_beats('madmom')
		self.write_downbeats('qm')
		self.write_downbeats('madmom')
		self.write_csv_format('qm')
		self.write_csv_format('essentia')
		self.write_csv_format('madmom')
	
	def load_estimates(self, ind):
		for extractor_type in ['qm','essentia','madmom']:
			csv_path = self.make_csv_path(extractor_type)
			tmp_rhythm_data = self.read_csv_format(csv_path)
			self.rhythm_data[extractor_type] = tmp_rhythm_data

	def evaluate_estimates(self, ind):
		# beat_scores = []
		# dbeat_scores = []
		ref_beats = self.beats['true'].beat_onset
		ref_dbeats = self.beats['true'].beat_onset[self.beats['true'].beat==1]
		# Evaluate beats
		b_scores = []
		for ext in ['qm','madmom','essentia']:
			est_beats = self.beats[ext].beat_onset
			b_scores += [self.get_scores(ref_beats, est_beats)]
		# Evaluate downbeats:
		db_scores = []
		for ext in ['qm','madmom']:
			est_dbeats = np.array(self.beats[ext].beat_onset[self.beats[ext].beat==1])
			db_scores += [self.get_scores(ref_dbeats, est_dbeats)]	
		return b_scores, db_scores
	
	def eval_downbeats(self, ind):
		est_beat_opts = [self.qm_dt, self.mm_dt]
		ref_beats = self.true_dt
		scores = [self.get_scores(np.array(ref_beats), np.array(est_beat_opts[i])) for i in range(len(est_beat_opts))]	
		return scores	
	
	def get_scores(self, ref_beats, est_beats):
		f_measure = mir_eval.beat.f_measure(ref_beats, est_beats)
		cemgil = mir_eval.beat.cemgil(ref_beats, est_beats)
		goto = mir_eval.beat.goto(ref_beats, est_beats)
		p_score = mir_eval.beat.p_score(ref_beats, est_beats)
		continuity = mir_eval.beat.continuity(ref_beats, est_beats)
		information_gain = mir_eval.beat.information_gain(ref_beats, est_beats)
		scores_list = [f_measure, goto, p_score, information_gain] + list(cemgil) + list(continuity)
		return scores_list
	
	# @staticmethod
	# def rhythm_beat_times(rhythm_data):
	# 	return np.array(rhythm_data['onset'])
	#
	# @staticmethod
	# def rhythm_downbeat_times(rhythm_data, phase=1):
	# 	return np.array(rhythm_data['onset'][rhythm_data['beat']==phase])
	
	# def match_octave_phase(self, rhythm_data, rhythm_true):
		
