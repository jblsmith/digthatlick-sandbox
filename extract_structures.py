from matplotlib import pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
data_dir_opts = ["/Users/jordan/Documents/data/WeimarJazzDatabase","/home/jordansmith/Documents/data/WeimarJazzDatabase"]
for data_dir in data_dir_opts:
	if os.path.exists(data_dir):
		data_dir = data_dir

solo_structure_folder = data_dir + "/annotations/solo_structure/"
track_structure_filder = data_dir + "/annotations/track_structure/"
import tracker
reload(tracker)
beat = tracker.Beat()
beat.load_metadata('melody')
beat.load_metadata('sections')
# beat.load_metadata('notes')
beat.load_metadata('solo_info')
beat.load_metadata('transcription_info')
beat.load_metadata('track_info')
# beat.load_metadata('compositions_info')
beat.load_metadata('beats',set_table_name='md_beats')


sec_types = np.unique(beat.sections.type)
melids = np.unique(beat.transcription_info.melid)
for melid in melids:
	for sec_type in sec_types:
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


# Use MSAF to analyze structure of a song.
import msaf
import librosa

trackid=6
track_audiopath = beat.audio_orig_dir + beat.track_info.loc[beat.track_info.trackid==trackid]["filename_track"].iloc[0]+".wav"
boundaries, labels = msaf.process(track_audiopath)


sonified_file = "my_boundaries.wav"
sr = 44100
boundaries, labels = msaf.process(track_audiopath, sonify_bounds=True, 
                                  out_bounds=sonified_file, out_sr=sr)

melids = np.unique(beat.sections.melid.loc[beat.sections.trackid==trackid])

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


