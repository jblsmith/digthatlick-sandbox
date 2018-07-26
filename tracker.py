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
import re
import scipy as sp
import vamp

# A RhythmData object contains a two-level rhythm description: beat and bar (which is one level beyond beat). You could use another RhythmData object to describe the sub-beat or super-bar scales.
# It has three main attributes, all numpy arrays:
# - beat_onset : real-valued beat onset positions
# - beat       : integer beat indices (the counts within the bar)
# - bar        : integer bar indices (the bar numbers)
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
			# self.beat = [range(time_sig)[np.mod(i,time_sig)]+1 for i in range(len(self.beat_onset))]
			self.beat = self.project_beats(len(self.beat_onset), 1, time_sig)
		
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
	
	def df(self):
		return pd.DataFrame({'bar':self.bar, 'beat':self.beat, 'onset':self.beat_onset})
	
	def head(self, n=10):
		return self.df().head(n)
	
	def summary(self):
		print "Median beat period: " + str(np.median(np.diff(self.beat_onset)))
		print "Number of beats:    " + str(len(self.beat_onset))
		print "First downbeat:     " + str(self.beat_onset[list(self.beat).index(1)])
	
	def period(self):
		return np.median(np.diff(self.beat_onset))
	
	def phase(self):
		return np.mean(np.mod(self.beat_onset, self.period()))
		# return self.beat_onset[list(self.beat).index(1)]
	
	def downbeats(self):
		return self.beat_onset[self.beat==1]
	
	def __repr__(self):
		return "<Rhythm period:%s nbeats:%s phase:%s>" % (self.period(),len(self.beat_onset), self.phase())
	
	def shift_beats(self, offset):
		# Obtain an array with the beats shifted forwards or backwards.
		# For example, if the beats are [1,2,3,4,1,2,3,4], you can shift them forwards by 2 to get:
		# 	[3,4,1,2,3,4,1,2]
		# or backwards by 1 to get:
		#	[4,1,2,3,4,1,2,3]
		if offset>0:
			new_beat = np.concatenate((self.beat[offset:], self.project_beats(offset,self.beat[-1]+1,np.max(self.beat))))
		elif offset<0:
			new_beat = np.concatenate((self.project_beats(-offset,self.beat[0],np.max(self.beat), forwards=False), self.beat[:offset] ))
		elif offset==0:
			new_beat = self.beat
		new_rhythm = RhythmData(beat_onset = self.beat_onset, beat=new_beat)
		# bars = np.cumsum(np.array(self.beat[1:])<np.array(self.beat[:-1]))
		# self.bar = np.concatenate(([0], bars))
		return new_rhythm
	
	def project_beats(self, n_beats, first_beat=1, max_beats=4, forwards=True):
		# Continue a sequence of beats forwards, or backwards, from a given starting point.
		# E.g., if the given beat sequence is: [1,2,3,4,1,2,3,4,1]
		if forwards:
			return np.array([np.mod(i+(first_beat-1),max_beats)+1 for i in range(n_beats)]).astype(int)
		if not forwards:
			return np.array([np.mod(i+(first_beat-1),max_beats)+1 for i in np.arange(-n_beats,0)]).astype(int)
	
	def inject_beats(self, submeter=3):
		# Assume the beats are downbeats and inject beats with a particular spacing (submeter) in between.
		new_onsets = np.concatenate([np.linspace(self.beat_onset[i],self.beat_onset[i+1],submeter, endpoint=False) for i in range(len(self.beat_onset)-1)])
		new_onsets = np.append(new_onsets, self.beat_onset[-1])
		downbeats = np.array([onset in self.beat_onset for onset in new_onsets])
		new_rhythm = RhythmData(beat_onset = new_onsets, bar=np.cumsum(downbeats))
		return new_rhythm

	def downscale_meter(self, subdivisions=2, downbeat_indices=[1,3]):
		# Same as "inject beats", but don't assume that all original beats are downbeats. Instead, allow any beat with a beat index in downbeat_indices to be a downbeat..
		new_onsets = np.concatenate([np.linspace(self.beat_onset[i],self.beat_onset[i+1],subdivisions, endpoint=False) for i in range(len(self.beat_onset)-1)])
		new_onsets = np.append(new_onsets, self.beat_onset[-1])
		orig_beats_that_are_downbeats = [self.beat_onset[i] for i in range(len(self.beat_onset)) if self.beat[i] in downbeat_indices]
		downbeats = np.array([onset in orig_beats_that_are_downbeats for onset in new_onsets])
		new_rhythm = RhythmData(beat_onset = new_onsets, bar=np.cumsum(downbeats))
		return new_rhythm
	
	def superject_beats(self, supermeter, phase_offset=0):
		# Count the downbeats as beats and assume a hypermeter with a particular phase.
		# Phase offset defines what the beat index of the first beat should be.
		new_onsets = np.array([self.beat_onset[i] for i in range(len(self.beat_onset)) if self.beat[i]==1])
		# bars = np.zeros(len(new_onsets))
		# bars[phase] = 1
		bars = np.cumsum(np.array([np.mod(i,supermeter)==0 for i in range(len(new_onsets))]))
		new_rhythm = RhythmData(beat_onset = new_onsets, bar=bars)
		new_rhythm = new_rhythm.shift_beats(offset=phase_offset)
		# new_rhythm.beat = new_rhythm.get_shifted_beats(offset=phase_offset)
		# new_rhythm.bar = []
		# new_rhythm.infer_missing_data()
		return new_rhythm
	
	def upscale_meter(self, supermeter=4, beat_indices=[1,3], phase_offset=0):
		new_onsets = [x[0] for x in zip(self.beat_onset, self.beat) if x[1] in beat_indices]
		bars = np.cumsum(np.array([np.mod(i,supermeter)==0 for i in range(len(new_onsets))]))
		new_rhythm = RhythmData(beat_onset = new_onsets, bar=bars)
		new_rhythm = new_rhythm.shift_beats(offset=phase_offset)
		return new_rhythm
	

class Beat(object):

	# Initial management:
	# - Setting up paths
	# - Loading beats
	# - Loading audio

	def __init__(self):
		data_dir_opts = ["/Users/jordan/Documents/data/WeimarJazzDatabase","/home/jordansmith/Documents/data/WeimarJazzDatabase"]
		for data_dir in data_dir_opts:
			if os.path.exists(data_dir):
				self.data_dir = data_dir
		# self.data_dir = 
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
		self.load_metadata("track_info")
		self.load_metadata("solo_info")
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
	
	def make_beat_path(self, beat_string, extractor_type = 'madmom'):
		# if beat_type == 0:
		# 	beat_string = 'beats'
		# elif beat_type == 1:
		# 	beat_string = 'downbeats'
		return self.est_dir + extractor_type + "/" + str(self.ind) + "-" + beat_string + ".txt"
	
	def make_csv_path(self, extractor_type = 'madmom'):
		return self.est_dir + extractor_type + "/" + str(self.ind) + ".csv"

	# FIXME: since some of this information is needed for meta-analysis, you must re-create it --- just, ideally, from the transcription_info and song_info DB contents..
	
	def load_metadata(self, table_field="track_info"):
		track_info = self.db[table_field].all()
		data_list_of_lists = [row for row in track_info]
		if len(data_list_of_lists)==0:
			print "Sorry, there is no data in that table."
		else:
			colnames = data_list_of_lists[0].keys()
			table_as_df = pd.DataFrame(data=data_list_of_lists, columns=colnames)
			setattr(self,table_field,table_as_df)
		# self.track_info = table_as_df
		# import parsedatetime
		# self.track_info['year']
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
	
	def load_true_beats_and_downbeats(self, ind=None):
		if ind is None:
			ind=self.ind
		relevant_beats = self.db['beats'].find(melid=ind)
		colnames = relevant_beats.keys
		data_list_of_lists = [row for row in relevant_beats]
		if len(data_list_of_lists)==0:
			print "ERROR: There is no such melid!"
			return
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

	def read_csv_format(self, beat_file_path):
		csv_data = pd.read_csv(beat_file_path,header=0)
		csv_data[['bar','beat']] = csv_data[['bar','beat']].astype(int)
		csv_data['onset'] = csv_data['onset'].astype(float)
		return csv_data
	
	def write_csv_format(self, extractor):
		write_path = self.make_csv_path(extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			self.beats[extractor].df().to_csv(filehandle, index=False)
	
	def write_beats(self, extractor):
		write_path = self.make_beat_path(beat_string = 'beats', extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			filehandle.write("\n".join(list(self.beats[extractor].beat_onset.astype(str))))
	
	def write_downbeats(self, extractor):
		write_path = self.make_beat_path(beat_string = 'downbeats', extractor_type = extractor)
		with open(write_path,'w') as filehandle:
			# self.rhythm_data[extractor].to_csv(filehandle)
			filehandle.write("\n".join(list(self.beats[extractor].beat_onset[self.beats[extractor].beat==1].astype(str))))
	
	# Shorthand routines for running / writing / loading all estimates
	def run_all_estimates(self):
		for extractor_type in ['qm','essentia','madmom']:
			self.estimate_beats(extractor_type)
		# self.estimate_beats(extractor_type='qm')
		# self.estimate_beats(extractor_type='essentia')
		# self.estimate_beats(extractor_type='madmom')
	
	def write_all_rhythms(self):
		for extractor_type in list(set.intersection(set(self.beats.keys()), set(['qm','madmom']))):
			self.write_beats(extractor_type)
			self.write_downbeats(extractor_type)
			self.write_csv_format(extractor_type)
		for extractor_type in list(set.intersection(set(self.beats.keys()), set(['essentia']))):
			self.write_beats(extractor_type)
			self.write_csv_format(extractor_type)
		# self.write_beats('qm')
		# self.write_beats('essentia')
		# self.write_beats('madmom')
		# self.write_downbeats('qm')
		# self.write_downbeats('madmom')
		# self.write_csv_format('qm')
		# self.write_csv_format('essentia')
		# self.write_csv_format('madmom')
	
	def load_estimates(self, methods=['qm','essentia','madmom']):
		for extractor_type in methods:
			csv_path = self.make_csv_path(extractor_type)
			tmp_rhythm_data = self.read_csv_format(csv_path)
			# self.beats[extractor_type] = tmp_rhythm_data
			self.beats[extractor_type] = RhythmData(beat_onset=tmp_rhythm_data.onset,bar=tmp_rhythm_data.bar,beat=tmp_rhythm_data.beat)

	def evaluate_estimates(self):
		# beat_scores = []
		# dbeat_scores = []
		ref_beats = self.beats['true'].beat_onset
		ref_dbeats = self.beats['true'].beat_onset[self.beats['true'].beat==1]
		# Evaluate beats
		# b_scores = []
		# for ext in ['qm','madmom','essentia']:
		# 	est_beats = self.beats[ext].beat_onset
		# 	b_scores += [get_scores(ref_beats, est_beats)]
		b_scores = {ext: get_scores(ref_beats, self.beats[ext].beat_onset) for ext in ['qm','madmom','essentia'] if ext in self.beats.keys()}
		db_scores = {ext: get_scores(ref_dbeats, np.array(self.beats[ext].beat_onset[self.beats[ext].beat==1])) for ext in ['qm','madmom'] if ext in self.beats.keys()}
		# Evaluate downbeats:
		# db_scores = []
		# for ext in ['qm','madmom']:
		# 	est_dbeats = np.array(self.beats[ext].beat_onset[self.beats[ext].beat==1])
		# 	db_scores += [get_scores(ref_dbeats, est_dbeats)]
		return b_scores, db_scores
	
	# def eval_downbeats(self):
	# 	est_beat_opts = [self.qm_dt, self.mm_dt]
	# 	ref_beats = self.true_dt
	# 	scores = [get_scores(np.array(ref_beats), np.array(est_beat_opts[i])) for i in range(len(est_beat_opts))]
	# 	return scores


# # # # # 
# # # # # Evaluation scripts
# # # # # 

def get_scores(ref_beats, est_beats):
	f_measure = mir_eval.beat.f_measure(ref_beats, est_beats)
	cemgil = mir_eval.beat.cemgil(ref_beats, est_beats)
	goto = mir_eval.beat.goto(ref_beats, est_beats)
	p_score = mir_eval.beat.p_score(ref_beats, est_beats)
	continuity = mir_eval.beat.continuity(ref_beats, est_beats)
	information_gain = mir_eval.beat.information_gain(ref_beats, est_beats)
	scores_list = [f_measure, goto, p_score, information_gain] + list(cemgil) + list(continuity)
	return scores_list

def evaluate_rdata(ref_data, est_data):
	# Evaluate beats
	ref_beats = ref_data.beat_onset
	est_beats = est_data.beat_onset
	beat_eval = get_scores(ref_beats, est_beats)
	# Evaluate downbeats
	ref_beats = ref_data.superject_beats(supermeter=4).beat_onset
	est_beats = est_data.superject_beats(supermeter=4).beat_onset
	downbeat_eval = get_scores(ref_beats, est_beats)
	return beat_eval, downbeat_eval		

def find_best_match(rhythm, true_rhythm, n_levels=[2,2], meter=[2,2]):
	# meter = [submeter, supermeter]
	# n_levels = [n_sublevels, n_superlevels]
	#
	# First, create a set of rhythms at different hierarchical levels above and below the main rhythm:
	rhythms = [(rhythm,"normal")]
	for i in range(n_levels[0]):
		rhythms.insert(0, (rhythms[0][0].downscale_meter(subdivisions=meter[0], downbeat_indices=[1,3]), "down"+str(i+1)))
	for i in range(n_levels[1]):
		# rhythms.append(rhythms[-1].superject_beats(supermeter=meter[1]))
		rhythms.append((rhythms[-1][0].upscale_meter(supermeter=4, beat_indices=[1,3], phase_offset=0), "up"+str(i+1)))
	# Second, for each rhythm, shift it to its various phases:
	rhy_matrix = [ [(rh[0].shift_beats(i),rh[1]+"_shift"+str(i)) for i in range(4)] for rh in rhythms]
	grades = np.zeros((len(rhy_matrix),len(rhy_matrix[0]),2,10))
	for i in range(len(rhy_matrix)):
		for j in range(len(rhy_matrix[i])):
			grades[i,j,:,:] = np.array(evaluate_rdata(true_rhythm,rhy_matrix[i][j][0]))
	best_beat = np.unravel_index(grades[:,:,0,0].argmax(), grades[:,:,0,0].shape)
	best_downbeat = np.unravel_index(grades[:,:,1,0].argmax(), grades[:,:,1,0].shape)
	return rhy_matrix, grades, best_beat, best_downbeat

# # # # #
# # # # # Full dataset analysis routines
# # # # #

def extract_all_beats(methods=['qm','madmom','essentia'], trackids=None):
	beat = Beat()
	if trackids is None:
		# trackids = beat.track_info.trackid
		trackids = beat.solo_info.melid
	for trackid in trackids:
		print "Doing " + str(trackid)
		try:
			# Load audio
			beat.set_index(trackid)
			# Run estimators
			beat.run_all_estimates()
			# Write outputs to files
			beat.write_all_rhythms()
		except KeyboardInterrupt:
			raise
		except:
			print "Failed for " + str(trackid)

def evaluate_all_beats(methods=['qm','madmom','essentia'], trackids=None, level='beats'):
	beat = Beat()
	if trackids is None:
		# trackids = beat.track_info.trackid
		trackids = beat.solo_info.melid
	overall_results = np.zeros((len(trackids),len(methods),2,5,4))
	period_errs = []
	phase_errs = []
	for ti,trackid in enumerate(trackids):
		print "Doing " + str(trackid)
		beat.ind = trackid
		beat.load_true_beats_and_downbeats()
		beat.load_estimates()
		true = beat.beats['true']
		errs = [get_phase_and_period_error(beat.beats[method], true, level) for method in methods]
		period_errs += [zip(*errs)[0]]
		phase_errs += [zip(*errs)[1]]
	return period_errs, phase_errs
		# for mi,method in enumerate(methods):
		# 	if method in beat.beats.keys():
		# 		est = beat.beats[method]
		# 		rhy_matrix, grades, best_beat, best_downbeat = find_best_match(est, true)
		# 		overall_results[ti,mi,0,:,:] = grades[:,:,0,0]
		# 		overall_results[ti,mi,1,:,:] = grades[:,:,1,0]
	# return overall_results

def get_phase_and_period_error(est_rhythm, true_rhythm, level='beats'):
	if level=='beats':
		est = est_rhythm.beat_onset
		true = true_rhythm.beat_onset
	elif level=='downbeats':
		est = est_rhythm.downbeats()
		true = true_rhythm.downbeats()
	else:
		print "Error, invalid level. Choose 'beats' or 'downbeats'."
	est = est[est-np.min(true)>-1]
	est = est[est-np.max(true)<1]
	est_period = np.median(np.diff(est))
	true_period = np.median(np.diff(true))
	period_info = np.array([est_period, true_period, est_period/true_period, true_period/est_period])
	distances = np.array([true-est[i] for i in range(len(est))])
	true_to_est = np.min(np.abs(distances),axis=0)
	est_to_true = np.min(np.abs(distances),axis=1)
	# phase_info = np.array([np.median(est_to_true)/est_period, np.median(true_to_est)/true_period,
	# 						np.median(est_to_true)/true_period, np.median(true_to_est)/est_period])
	phase_info = np.array([np.median(est_to_true), np.median(true_to_est),
							np.median(est_to_true)/true_period, np.median(true_to_est)/est_period])
	# phase_info = np.array((tmp_phase_info, tmp_phase_info / est_period, tmp_phase_info / true_period))
	# tmp_phase_info = np.array([np.median(est_to_true), np.median(true_to_est)])
	# phase_info = np.array((tmp_phase_info, tmp_phase_info / est_period, tmp_phase_info / true_period))
	return period_info, phase_info

# Or should I somehow look at a *histogram* of distances from est-to-true to figure out what's up?
# NO--- want to look at phase alone, not consider the error in tempo. So to look at phase alone, just want to find out how far from the true beats the estimated ones are.