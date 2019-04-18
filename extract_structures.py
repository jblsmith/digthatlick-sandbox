# Preamble
from matplotlib import pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
import os
data_dir_opts = ["/Users/jordan/Documents/data/WeimarJazzDatabase","/home/jordansmith/Documents/data/WeimarJazzDatabase"]
for data_dir in data_dir_opts:
	if os.path.exists(data_dir):
		data_dir = data_dir

solo_structure_folder = data_dir + "/annotations/solo_structure/"
track_structure_filder = data_dir + "/annotations/track_structure/"
import tracker
import importlib.reload as reload
reload(tracker)
beat = tracker.Beat()


#
#
# Metadata to prepare structure description extraction
#
#
beat.load_metadata('solo_info')
beat.load_metadata('transcription_info')
beat.load_metadata('track_info')
beat.load_metadata('melody')
beat.load_metadata('sections')
# beat.load_metadata('notes')
# beat.load_metadata('compositions_info')
beat.load_metadata('beats',set_table_name='md_beats')


# Get structure by looking at section boundaries defined by melody events.
def get_structure(beat,sec_type,melid):
	for md_field in ['sections','melody','solo_info','transcription_info','track_info']:
		if md_field not in beat.__dict__.keys():
			beat.load_metadata(md_field)
	if 'md_beats' not in beat.__dict__.keys():
		beat.load_metadata('beats',set_table_name='md_beats')
	songsecs = beat.sections.loc[(beat.sections.melid==melid) & (beat.sections.type==sec_type)]
	songmels = beat.melody.loc[beat.melody.melid==melid]
	onset_times = [songmels.iloc[x]["onset"] for x in songsecs.start]
	offset_times = [songmels.iloc[x]["onset"] + songmels.iloc[x]["duration"] for x in songsecs.end]
	# Note: these are times given with respect to the solo, not the original recording.
	solo_filename = beat.transcription_info[beat.transcription_info.melid==melid]["filename_solo"].iloc[0]
	output_filename = "_".join(["structure",str(melid),sec_type,"-",solo_filename]) + ".txt"
	output_df = pd.DataFrame({"onset":onset_times, "offset":offset_times, "label":songsecs.value})
	output_df = output_df[["onset","offset","label"]]
	# output_df.index = range(len(onset_times))
	output_df.to_csv(solo_structure_folder+output_filename,header=False,index=False,sep="\t")		

# Get structure by looking at the chorus_id values in the beats table.
melid = 1
songsecs = beat.sections.loc[beat.sections.melid==melid]
songbeats = beat.md_beats.loc[beat.md_beats.melid==melid]
chorus_starts = np.insert(np.diff(songbeats.chorus_id),0,1)
# songbeats.loc[chorus_starts==1]
output_df = songbeats.loc[chorus_starts==1, ["onset","form","chorus_id"]]
last_beat = songbeats.iloc[-1]["onset"] + np.median(np.diff(songbeats.onset))
offsets = list(output_df.onset.iloc[1:]) + [last_beat]
output_df["offset"] = offsets
solo_offset = beat.transcription_info[beat.transcription_info.melid==melid]["solostart_sec"].iloc[0]
# output_df.onset += solo_offset
output_filename = "_".join(["structure",str(melid),sec_type,"-",solo_filename,'fromBeats']) + ".txt"
output_df.to_csv(solo_structure_folder+output_filename,header=False,index=False,sep="\t")


sec_types = np.unique(beat.sections.type)
melids = np.unique(beat.transcription_info.melid)
for melid in melids:
	for sec_type in sec_types:
		get_structure(beat,sec_type,melid)

		# Get structure
		songsecs = beat.sections.loc[(beat.sections.melid==melid) & (beat.sections.type==sec_type)]
		songmels = beat.melody.loc[beat.melody.melid==melid]
		onset_times = [songmels.iloc[x]["onset"] for x in songsecs.start]
		offset_times = [songmels.iloc[x]["onset"] + songmels.iloc[x]["duration"] for x in songsecs.end]
		# Note: these are times given with respect to the solo, not the original recording.
		solo_offset = beat.transcription_info[beat.transcription_info.melid==melid]["solostart_sec"].iloc[0]
		solo_filename = beat.transcription_info[beat.transcription_info.melid==melid]["filename_solo"].iloc[0]
		output_filename = "_".join(["structure",str(melid),sec_type,"-",solo_filename]) + ".txt"
		output_df = pd.DataFrame({"onset":onset_times, "offset":offset_times, "label":songsecs.value})
		output_df = output_df[["onset","offset","label"]]
		# output_df.index = range(len(onset_times))
		output_df.to_csv(solo_structure_folder+output_filename,header=False,index=False,sep="\t")

trackids = np.unique(beat.solo_info.trackid)
beat.sections.insert(loc=len(beat.sections.columns),column="trackid",value=0)
for melid in melids:
	beat.sections.loc[beat.sections.melid==melid, "trackid"] = beat.transcription_info.trackid[beat.transcription_info.melid==melid].iloc[0]


for trackid in trackids:
	for sec_type in sec_types:
		songsecs = beat.sections.loc[(beat.sections.trackid==trackid) & (beat.sections.type==sec_type)]
		dfs = []
		if len(np.unique(songsecs.melid))>1:
			print "More than one solo for trackid = " +str(trackid)
		for melid in np.unique(songsecs.melid):
			songsecs_tmp = songsecs.loc[songsecs.melid==melid]
			songmels = beat.melody.loc[beat.melody.melid==melid]
			onset_times = [songmels.iloc[x]["onset"] for x in songsecs_tmp.start]
			offset_times = [songmels.iloc[x]["onset"] + songmels.iloc[x]["duration"] for x in songsecs_tmp.end]
			# Note: these are times given with respect to the solo, not the original recording.
			solo_offset = beat.transcription_info[beat.transcription_info.melid==melid]["solostart_sec"].iloc[0]
			onset_times += solo_offset
			offset_times += solo_offset
			output_df = pd.DataFrame({"onset":onset_times, "offset":offset_times, "label":songsecs_tmp.value})
			output_df = output_df[["onset","offset","label"]]
			dfs += [output_df]
		output_df = pd.concat(dfs)
		track_filename = beat.track_info[beat.track_info.trackid==trackid]["filename_track"].iloc[0]
		output_filename = "_".join(["structure",str(trackid),sec_type,"-",track_filename]) + ".txt"
		# output_df.index = range(len(onset_times))
		output_df.to_csv(track_structure_filder+output_filename,header=False,index=False,sep="\t")		





# Actually, let's use a new dataset-driven approach to saving the features, maybe!
import dataset
import librosa
beat = tracker.Beat()
beat.load_metadata('track_info')
trackids = beat.track_info.trackid
trackid = trackids[0]
track_audiopath = beat.audio_orig_dir + beat.track_info.loc[beat.track_info.trackid==trackid]["filename_track"].iloc[0]+".wav"
# Extract hpcp feats
y, sr = librosa.load(track_audiopath)
cqt = librosa.feature.chroma_cqt(y, sr=sr, n_chroma=12, n_octaves=6)

db = dataset.connect('sqlite:///mydatabase.db')
db = dataset.connect()
table = db['song']
table.insert(dict(cqt=cqt))


# Sonic annotator interlude:
sonic-annotator -s vamp:qm-vamp-plugins:qm-chromagram:chromagram > '/home/jordansmith/Dropbox/Ircam/code/n3s/chromagram.n3'
sonic-annotator -t '/home/jordansmith/Dropbox/Ircam/code/n3s/chromagram.n3' -r '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/wav_orig/' --csv-basedir '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/chromagrams/' --csv-fill-ends -w csv

sonic-annotator -s vamp:qm-vamp-plugins:qm-mfcc:coefficients > '/home/jordansmith/Dropbox/Ircam/code/n3s/mfcc.n3'
sonic-annotator -t '/home/jordansmith/Dropbox/Ircam/code/n3s/mfcc.n3' -r '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/wav_orig/' --csv-basedir '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/mfccs/' --csv-fill-ends -w csv

# Extract for solo files too
sonic-annotator -t '/home/jordansmith/Dropbox/Ircam/code/n3s/chromagram.n3' -r '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/wav_solo/' --csv-basedir '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/chromagrams_solo/' --csv-fill-ends -w csv
sonic-annotator -t '/home/jordansmith/Dropbox/Ircam/code/n3s/mfcc.n3' -r '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/wav_solo/' --csv-basedir '/home/jordansmith/Documents/data/WeimarJazzDatabase/audio/mfccs_solo/' --csv-fill-ends -w csv

# Now we can quickly load anything.
trackid=1
time_c, chroma = fetch_feat(beat,4,'chroma')
time_m, mfcc = fetch_feat(beat,4,'mfcc')


#
#
# Look at whether feature consistency tells us anything about phase probability. Focus on downbeats, which are the least usable as is.
#
#

beat = tracker.Beat()
beat.load_metadata('solo_info')
beat.load_metadata('transcription_info')
beat.load_metadata('track_info')
scores = []
stds = []
melids = []
for melid in range(1,523):
	print melid
	# Analyze the song: get downbeats with qm and/or madmom.
	beat.ind = melid
	beat.load_true_beats_and_downbeats()
	beat.load_estimates()
	true = beat.beats['true']
	methods=['qm','madmom']
	errs = [tracker.get_phase_and_period_error(beat.beats[method], true, 'downbeats') for method in methods]
	per = np.array(zip(*errs)[0])
	pha = np.array(zip(*errs)[1])
	
	# Generate new time series with alternative hypotheses for phase.
	x1 = beat.beats['madmom']
	x2 = x1.shift_beats(1)
	x3 = x1.shift_beats(2)
	x4 = x1.shift_beats(3)
	db1, db2, db3, db4 = [b.downbeats() for b in [x1,x2,x3,x4]]
	score = [tracker.get_phase_and_period_error(x, beat.beats['true'], level='downbeats')[1][2] for x in [x1,x2,x3,x4]]
	# tracker.get_phase_and_period_error(x2, beat.beats['true'], level='downbeats')[1][2]
	# tracker.get_phase_and_period_error(x3, beat.beats['true'], level='downbeats')[1][2]
	# tracker.get_phase_and_period_error(x4, beat.beats['true'], level='downbeats')[1][2]
	
	# Collect chroma and mfccs.
	time_c, chroma = tracker.fetch_feat(beat,melid=melid,feature='chroma')
	# time_c, chroma = tracker.fetch_feat(beat,melid=melid,feature='mfcc')
	# time_m, mfcc = tracker.fetch_feat(beat,melid=melid,feature='mfcc')
	if time_c is not None:
		# Aggregate features over different time series hypotheses.
		# c1 =librosa.util.sync(chroma, db1)
		std1 = manual_sync(chroma, time_c, db1, 'std')
		std2 = manual_sync(chroma, time_c, db2, 'std')
		std3 = manual_sync(chroma, time_c, db3, 'std')
		std4 = manual_sync(chroma, time_c, db4, 'std')
		
		scores += [score]
		stds += [[np.mean(s) for s in [std1,std2,std3,std4]]]
		melids += [melid]


# Remember: 'scores' has the (est-to-true-distance)/true_period. These will be around 0 when correct, around .25 when off by a beat, .5 when off by two beats (half a downbeat).
# We want to know whether the low-stdev windows correspond to the correct windows. The winners (lewst score) are:
winners = np.argmin(scores,axis=1)
# The calmest (lowest STDEV) versions of the downbeat train are:
calmers = np.argmin(stds,axis=1)
# The opposite of that are:
wilders = np.argmax(stds,axis=1)
np.mean(winners==calmers)
np.mean(winners==wilders)
np.mean(winners==calmers+1)

calmers = [np.argmin(row) for row in stds]

def manual_sync(feat, times, new_times, func='mean'):
	new_feat = np.zeros((len(new_times)-1, feat.shape[1]))
	indices = [np.where( (times>=new_times[i]) & (times<new_times[i+1]))[0] for i in range(len(new_times)-1)] 
	for t in range(len(new_times)-1):
		new_feat[t,:] = getattr(np, func)(feat[indices[t],:], axis=0)
		# new_feat[t,:] = np.mean(feat[indices[t],:],axis=0)
	return new_feat

# Visualize results somehow, and discover a trend!


#agg1 = manual_sync(chroma, time_c, db1, 'mean')
#agg2 = manual_sync(chroma, time_c, db2, 'mean')
#agg3 = manual_sync(chroma, time_c, db3, 'mean')
#agg4 = manual_sync(chroma, time_c, db4, 'mean')


melid=5
beat.ind = melid
beat.load_true_beats_and_downbeats()
beat.load_estimates()
# Collect chroma and mfccs.
time_c, chroma = tracker.fetch_feat(beat,melid=melid,feature='chroma')
time_m, mfcc = tracker.fetch_feat(beat,melid=melid,feature='mfcc')
	

plt.subplot(1,4,1)
plt.imshow(std1)
plt.subplot(1,4,2)
plt.imshow(std2)
plt.subplot(1,4,3)
plt.imshow(std3)
plt.subplot(1,4,4)
plt.imshow(std4)








#
#
# Use MSAF to analyze structure of a song.
#
#
import msaf
msaf.config.dataset.audio_dir = "."
import librosa
beat = tracker.Beat()
beat.load_metadata('track_info')
trackids = beat.track_info.trackid
collection = data_dir + "/audio/tmp_wav_set/"
results = msaf.process(collection, n_jobs=1, boundaries_id="foote", feature='pcp')



trackid=6
track_audiopath = beat.audio_orig_dir + beat.track_info.loc[beat.track_info.trackid==trackid]["filename_track"].iloc[0]+".wav"
boundaries, labels = msaf.process(track_audiopath)


sonified_file = "my_boundaries.wav"
sr = 44100
boundaries, labels = msaf.process(track_audiopath, sonify_bounds=True, 
                                  out_bounds=sonified_file, out_sr=sr)

melids = np.unique(beat.sections.melid.loc[beat.sections.trackid==trackid])


boundaries, labels = msaf.process(track_audiopath, boundaries_id="foote", labels_id="fmc2d")

def write_estimate(boundaries, labels):
	tmp_df = pd.DataFrame({"onset":boundaries[:-1],"offset":boundaries[1:],"labels":labels})
	tmp_df = tmp_df[["onset","offset","labels"]]
	tmp_df.to_csv("./tmp_estimate.txt",header=False,index=False,sep="\t")


audio_file
print(boundaries)




# pseudocode:
Get everything related to one song:
melid=21
songinfo = beat.transcription_info.loc[beat.transcription_info.melid==melid]
print songinfo.transpose()
songsecs = beat.sections.loc[(beat.sections.melid==melid) & (beat.sections.type=="CHORUS")]
print songsecs
songbeats =beat.md_beats.loc[beat.md_beats.melid==melid]
songbeats.head(), songbeats.tail()
songmels = beat.melody.loc[beat.melody.melid==melid]
songmels.head(), songmels.tail()

The start and end of sections are given by "start" and "end" columns in SECTIONS table.
The values refer to the index into the table songmels, which lists all the melody notes.
These melody events contain:
	onset  pitch  duration  period  division  bar  beat  tatum  subtatum
So they contain redundant information about the bar and beat! Interesting. Of course, they might share the same bar and beat if they're super fast. 
So, the goal was to convert metadata about structure into SALAMI-like annotations. I want to thus have the onset and offset TIMES of the structural segments, but to do that I'll need to look up the corresponding NOTES, and the BEAT TIME of those notes. That's kind of dumb and roundabout! Why can't the sections refer to the beat times themselves? These are the more concrete, trustworthy times. Oh well. 'S alright.




secs = [songsecs.loc[songsecs.type==type] for type in np.unique(songsecs.type)]


Get a single timing table for each song.

beat start	beat end	bar number	beat index	phrase label	phrase index	...


