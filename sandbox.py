import tracker
reload(tracker)
self = tracker.Beat()
self.set_index(9)
self.estimate_beats('qm')
self.estimate_beats('essentia')
self.estimate_beats('madmom')
self.write_beats('qm')
self.write_beats('essentia')
self.write_beats('madmom')
self.write_downbeats('qm')
self.write_downbeats('madmom')


import tracker
beat = tracker.Beat()
for i in range(3,60):
	try:
		beat.set_index(i)
		beat.run_estimates()
	except:
		print "Failed for " + str(i)


reload(tracker)
beat = tracker.Beat()
score_list = []
for ind in range(10,44):
	beat.set_index(ind)
	beat.load_estimates(ind)
	scores = beat.eval_downbeats(ind)
	score_list += [np.array(scores)]

np.mean(np.array(score_list),axis=0)
np.max(np.array(score_list),axis=0)
np.median(np.array(score_list),axis=0)



reload(tracker)
beat = tracker.Beat()
allscore_list = []
for ind in range(8,44):
	beat.set_index(ind)
	beat.load_estimates(ind)
	scores = beat.evaluate_estimates(ind)
	allscore_list += [np.array(scores)]

np.mean(np.array(allscore_list),axis=0)
np.max(np.array(allscore_list),axis=0)
np.median(np.array(allscore_list),axis=0)

>>> np.mean(np.array(allscore_list),axis=0)
array([[0.14265863, 0.04651163, 0.12448969, 0.01422792],
       [0.14532461, 0.04651163, 0.12606321, 0.01527538],
       [0.15445101, 0.02325581, 0.13375799, 0.02363707],
       [0.03516958, 0.04651163, 0.13644896, 0.06500443],
       [0.05229624, 0.04651163, 0.13802641, 0.0858847 ]])
>>> np.max(np.array(allscore_list),axis=0)
array([[0.3235486 , 1.        , 0.41666667, 0.0389523 ],
       [0.32984293, 1.        , 0.4629156 , 0.05626468],
       [0.77749361, 1.        , 0.96428571, 0.34937435],
       [0.27627628, 1.        , 0.44086022, 0.27789929],
       [0.78787879, 1.        , 0.98      , 0.69459616]])
>>> np.median(np.array(allscore_list),axis=0)
array([[0.13303769, 0.        , 0.10869565, 0.01210271],
       [0.13390929, 0.        , 0.11402439, 0.0139886 ],
       [0.13425926, 0.        , 0.1225    , 0.01471179],
       [0.01967213, 0.        , 0.13253012, 0.05028261],
       [0.02380952, 0.        , 0.12790698, 0.06150333]])

{MEAN, MAX and MEDIAN}
				F-measure	Goto score	P-score		Info. gain
QM beats
ES beats
MM beats
QM downbeats
MM downbeats

f-measure	The F-measure of the beat sequence, where an estimated beat is considered correct if it is sufficiently close to a reference beat
Goto’s		A binary score which is 1 when at least 25% of the estimated beat sequence closely matches the reference beat sequence
P-score		McKinney’s P-score, which computes the cross-correlation of the estimated and reference beat sequences represented as impulse trains
Info		The Information Gain of a normalized beat error histogram over a uniform distribution

Conclusion: state of the art is indeed poor!
How poor, and why? --> Still TBD.
Note: median is far below the mean, so misses are total misses.
Madmom does the best overall. Its 'max' scores are usually very close to the GT, both for downbeat and beat.
	(But the mean madmom score is still only slightly better than the QM score, which is beatroot?)


Geoffroy asked : when beats are correct, what are the odds the downbeats are correct? (Even odds at that point are 1/4, at least, you would think)

Get baselines for structure segmentation / solo detection
Goal : finding non-melody examples.
Dogac : there are no labelled negative examples. We will need other annotations to get 'true negative' labels.
	Polina says : use backing tracks!
	Jordan : weak learning from non-positive labels?

Main meeting agenda to be set offline
April 23. 5pm here in Europe!


Beat tracking paper for ISMIR : submit an abstract in case!
If we have the time, get it done!
'Why is Jazz hard?'



Title:
Why Jazz is Hard: Assessing the state of the art for beat and melody analysis of Jazz music
Abstract:
Evaluations of beat tracking have generally focused on performance in popular music.
Jazz music is, generally, a much more challenging genre for beat tracking, as well as other basic MIR tasks like chord and melody transcription.
We evaluate a set of beat tracking algorithms on a large jazz corpus and demonstrate their shortcomings with extensive error analysis.
We find that not only is beat tracking and downbeat tracking in jazz more difficult overall, but that a smaller fraction of errors are explainable as octave errors.
Through our investigation of the results, we identify two musicological features of jazz music that lead to it being a challenging genre: first, the greater range of typical tempos, and second, the chaotic patterns of onsets.
We recommend a set of adaptations that future beat tracking algorithms could make to overcome these challenges.


# Beat and downbeat detection sandbox

#
# 
# 	Imports

import glob
import numpy as np
import scipy as sp
import librosa
import os.path
import madmom
import json
import vamp
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
import re

from beat import Beat

# 
# 
# Downloading all the data

# 1. Get metadata and get basenames from folder of SV files
data_dir = "/Users/jordan/Documents/data/WeimarJazzDatabase"
metadata_table_path = data_dir + "/annotations/weimar_contents.tsv"
metadata_table = open(metadata_table_path,'r').readlines()[1:]
song_indices = [line.split("\t")[0] for line in metadata_table]

sv_dir = data_dir + "/annotations/SV/"
sv_paths = glob.glob(sv_dir + "*.sv")
song_basenames = [re.sub("_FINAL.sv","",re.sub(sv_dir,"",path)) for path in sv_paths]
stripped_basenames = song_basenames[:]
for i in range(len(song_basenames)):
    if song_basenames[i][-2] == "-":
        stripped_basenames[i] = song_basenames[i][:-2]


orig_paths = glob.glob(data_dir+"/audio/wav_orig/*.wav")

# 2. Download melody and beat annotations
# They follow these conventions, where the final "/1" gives the song index:
for ind in song_indices:
    for ann_type in ["solo","beats"]:
        download_location = data_dir + "/annotations/" + ann_type + "/" + ind + ".csv"
        fp = open(download_location, "w")
        curl = pycurl.Curl()
        url_to_grab = "http://mir.audiolabs.uni-erlangen.de/jazztube/api/v1.0/export/" + ann_type + "/" + ind
        curl.setopt(pycurl.URL, url_to_grab)
        curl.setopt(pycurl.WRITEDATA, fp)
        curl.perform()
        curl.close()
        fp.close()

# 3. Download audio files
# Nevermind --- they made some zips available.
# for basename in stripped_basenames:
# "https://seafile.idmt.fraunhofer.de/d/bd238110f23e44a4bc0f/files/?p=/wav_orig/ZootSims_NightAndDay_Orig.wav"
# 'https://seafile.idmt.fraunhofer.de/d/bd238110f23e44a4bc0f/files/?p=/wav_orig/ZootSims_NightAndDay-2_Orig.wav'
# "https://seafile.idmt.fraunhofer.de/d/bd238110f23e44a4bc0f/files/?p=/wav_orig/ZootSims_NightAndDay_Orig.wav&dl=1"
# 
# download_location = data_dir + "/audio/wav_orig/" + basename + "_Orig.wav"
# fp = open(download_location, "wb")
# curl = pycurl.Curl()
# url_to_grab = "https://seafile.idmt.fraunhofer.de/d/bd238110f23e44a4bc0f/files/?p=/wav_orig/" + basename + "_Orig.wav&dl=1"
# curl.setopt(pycurl.URL, url_to_grab)
# curl.setopt(pycurl.WRITEDATA, fp)
# curl.perform()
# curl.close()
# fp.close()
#
#
# import requests
#
# r = requests.get(url_to_grab, auth=('myusername', 'mybasicpass'))
# print(r.text)
#
# data_dir = "/Users/jordan/Documents/data/WeimarJazzDatabase"



# 4. Run audio analysis 
# "/Users/jordan/Documents/data/WeimarJazzDatabase/audio/wav_orig/ArtBlakey_DownUnder_Orig.wav"

load audio
extract beats/db using qm vamp
extract using essentia library
extract using madmom
save all beats

separately,
load written estimate files
load annotation files
make comparisons using mir_eval
look to see what kinds of errors are made, whether they depend on the type of jazz or anything already parameterized

def extract_beats_three_ways(path):

orig_paths = glob.glob(data_dir+"/audio/wav_orig/*.wav")
sv_data_paths = [re.sub("_Orig.wav","_FINAL.sv",re.sub("/audio/wav_orig/","/SV/",path)) for path in orig_paths]
output_dump_dir = data_dir + "/algo_output"

song_index = 2
fs = 44100
audio_lib, sr_lib = librosa.core.load(orig_paths[song_index], sr=fs, mono=False)
signal_mono = librosa.to_mono(audio_lib)
# signal_mono_mm = madmom.audio.signal.remix(signal_mono.transpose(), num_channels=1)

bt1 = vamp.collect(signal_mono, sr_lib, 'qm-vamp-plugins:qm-barbeattracker')    # Beat and downbeat
# bt2 = vamp.collect(signal_mono, sr_lib, 'beatroot-vamp:beatroot')               # Beat
# bt3 = vamp.collect(signal_mono, sr_lib, 'qm-vamp-plugins:qm-tempotracker')      # Beat and tempo
t1,b1 = zip(*[(item['timestamp'], item['label']) for item in bt1['list']])
# t2,b2 = zip(*[(item['timestamp'], item['label']) for item in bt2['list']])
# t3,b3 = zip(*[(item['timestamp'], item['label']) for item in bt3['list']])


# essentia!
import essentia
import essentia.standard
import essentia.streaming
# loader = essentia.standard.MonoLoader(filename = orig_paths[song_index])
# audio_ess = loader()
beat_tracker = essentia.standard.BeatTrackerMultiFeature()
# ticks, confidence = beat_tracker(audio_ess)
ticks, confidence = beat_tracker(signal_mono)


mm_db_detection_function = madmom.features.beats.RNNDownBeatProcessor()(signal_mono)

downbeat_output = madmom.features.beats.DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(db_detection_function)

# Are essentia and librosa and madmom audios all truly different, or can I have them cooperate?


	def get_bars_and_beats_qm(self):
		import vamp
		if self.metre_qm is None:
			barbeattrack_output = vamp.collect(self.signal_mono, self.fs, 'qm-vamp-plugins:qm-barbeattracker')
			beat_times = np.array([float(item['timestamp']) for item in barbeattrack_output['list']])
			beat_metre = np.array([int(item['label']) for item in barbeattrack_output['list']])
			standard_beat_times_metre = np.array(zip(*(beat_times,beat_metre)))
			self.metre_qm = standard_beat_times_metre
	
	def get_bars_and_beats_mm(self):
		if self.metre_mm is None:
			self.prep_audio_for_madmom()
			# Beat tracking and downbeat tracking is hardcoded in 100ths of a second
			db_detection_function = madmom.features.beats.RNNDownBeatProcessor()(self.signal_mm_mono)
			downbeat_output = madmom.features.beats.DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(db_detection_function)
			self.metre_mm = downbeat_output
			# downbeats_mm_inSeconds = downbeat_output[np.where(downbeat_output[:,1]==1),0][0]
			# downbeats_mm_inframes = librosa.time_to_frames(downbeats_mm_inSeconds, sr=self.fs_mm)
			# add_downbeats_to_madmom_audio(audio, downbeats_mm_inSeconds)
			# return downbeat_output, downbeats_mm_inSeconds, downbeats_mm_inframes
	
	def prep_audio_for_madmom(self):
		if self.signal_mm is None:
			# I believe madmom requires this signal rate. There's something funny and hard-coded about sampling rates with madmom... so don't change this without testing carefully!
			self.fs_mm = 44100
			self.signal_mm = librosa.core.resample(self.signal, self.fs, self.fs_mm)
			self.signal_mm_mono = madmom.audio.signal.remix(self.signal_mm.transpose(), num_channels=1)



# OK, nevermind... the XML is fucked up somehow so can't read it. Never fucking mind.
# from lxml import etree
# parser = etree.XMLParser(recover=True)
# xmlstring = "\n".join(open(sv_data_paths[song_index],"r").readlines())
# etree.fromstring(xmlstring, parser=parser)
#
# parser = ET.XMLParser(encoding="utf-8")
# tree = ET.fromstring(xmlstring, parser=parser)
#
# tree = ET.parse(sv_data_paths[song_index])
# root = tree.getroot()


filename = "/Users/jordan/Documents/data/WeimarJazzDatabase/annotations/SV/ArtPepper_Anthropology_FINAL.sv"
wavfname = "/Users/jordan/Documents/data/WeimarJazzDatabase/audio/wav_solo/ArtPepper_Anthropology_Solo.wav"
# filename = "/Users/jordan/Documents/data/WeimarJazzDatabase/annotations/SV/ArtPepper_BluesForBlanche_FINAL.sv"
parameters, frames, durations, labels, values = extractSvlAnnotRegionFile(filename)

sve = py_sonicvisualiser.SVEnv(wavfname, filename)
sve = SVEnv()

# readSVLfiles.py

from xml.dom import minidom
import numpy as np

def extractSvlAnnotRegionFile(filename):
    """
    extractSvlAnnotRegionFile(filename)
    
    Extracts the SVL files (sonic visualiser)
    in this function, we assume annotation files
    for regions (generated by Sonic Visualiser 1.7.2,
    Regions Layer)
    
    Returns the following objects:
        parameters: copy-paste of the "header" of the SVL file,
        frames    : a numpy array containing the time
                    stamps of each frame, at the beginning
                    of the frame,
        durations : a numpy array containing the duration
                    of each frame,
        labels    : a dictionary with keys equal to the frame
                    number, containing the labels,
        values    : a dictionary with keys equal to the frame
                    number, containing the values.
    
    Note that this code does not parse the end of the xml file.
    The 'display' field is therefore discarded here.
    
    Numpy and xml.dom should be imported in order to use this
    function.
        
    Jean-Louis Durrieu, 2010
    firstname DOT lastname AT epfl DOT ch
    """
    ## Load the XML structure:
    dom = minidom.parse(filename)
    
    ## Keep only the data-tagged field:
    ##    note that you could also keep any other
    ##    field here. 
    dataXML = dom.getElementsByTagName('data')[0]
    
    ## XML structure for the parameters:
    parametersXML = dataXML.getElementsByTagName('model')[0]
    
    ## XML structure for the dataset field, containing the points: 
    datasetXML = dataXML.getElementsByTagName('dataset')[0]
    ## XML structure with all the points from datasetXML:
    pointsXML = datasetXML.getElementsByTagName('point')
    
    ## converting to a somewhat easier to manipulate format:
    ## Dictionary for the parameters:
    parameters = {}
    for key in parametersXML.attributes.keys():
        parameters[key] = parametersXML.getAttribute(key)
    
    ## number of points (or regions):
    nbPoints = len(pointsXML)
    
    ## number of attributes per point, not used here,
    ## but could be useful to check what type of SVL file it is?
    # nbAttributes = len(pointsXML[0].attributes.keys())
    
    ## Initialize the numpy arrays (frame time stamps and durations):
    frames = np.zeros([nbPoints], dtype=np.float)
    durations = np.zeros([nbPoints], dtype=np.float)
    ## Initialize the dictionaries (values and labels)
    values = {}
    labels = {}
    
    ## Iteration over the points:
    for node in range(nbPoints):
        ## converting sample to seconds for the time stamps and the durations:
        frames[node] = np.int(pointsXML[node].getAttribute('frame')) / np.double(parameters['sampleRate'])
        durations[node] = np.int(pointsXML[node].getAttribute('duration')) / np.double(parameters['sampleRate'])
        ## copy-paste for the values and the labels:
        values[node] = pointsXML[node].getAttribute('value')
        labels[node] = pointsXML[node].getAttribute('label')
        
    ## return the result:
    return parameters, frames, durations, labels, values
