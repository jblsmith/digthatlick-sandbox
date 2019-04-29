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
sys.path.append(os.path.expanduser("~/Dropbox/Ircam/code"))
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




all_names = []
all_types = []
all_forms = []
all_onsets = []
all_inds = []
part_lens = []
jaah_info = pd.read_csv("audio_paths_index.csv")
for ind in jaah_info.index:
	try:
		ann_json, part_names, part_types, part_forms, part_onsets, all_ons = parse_annotation(ind)
		all_names += [p.strip().lower() for p in part_names]
		all_types += [p.strip().lower() for sub_list in part_types for p in sub_list if p]
		all_forms += [p.strip().lower() for sub_list in part_forms for p in sub_list if p]
		all_onsets += part_onsets
		part_lens += [len(p) for p in all_ons]
		all_inds += [ind]*len(part_names)
	except:
		print("Failed on {0}".format(ind))

instrument_list = ['trumpet', 'trombone', 'clarinet', 'piano', 'sax', 'vocals', 'drums', 'horn', 'cornet', 'ensemble', 'vibraphone', 'guitar', 'banjo', 'scat', 'bass']
instrument_qualifiers = ['alto', 'tenor', 'bari', 'soprano', 'male', 'female']
func_list = ['solo', 'intro', 'outro', 'head', 'pickup', 'collective', 'improvisation', 'code', 'bridge']
block_words = ['-']

df = pd.DataFrame(data=all_names, columns=['part_name'])
df['ind'] = all_inds
df['onset'] = all_onsets
df['n_beats'] = part_lens
df['words'] = [[word.replace(",","") for word in title.split()] for title in all_names]
df['instruments'] = [[w for w in word_list if w in instrument_list] for word_list in df['words']]
df['inst_qualifiers'] = [[w for w in word_list if w in instrument_qualifiers] for word_list in df['words']]
df['funcs'] = [[w for w in word_list if w in func_list] for word_list in df['words']]
df['residual'] = [[w for w in word_list if w not in func_list+instrument_list+block_words+instrument_qualifiers] for word_list in df['words']]
np.unique(df.residual)

# For each song:
- run algorithm
- plot images of SSMs with segmentation superposed on one axis, estimated segmentation on the opposite axis
- 

-


# For each row:
# Do we know what instrument is played?
# Do we know what the section function is?
# Do we know what the section structure is?

# Is there any other unused information?

all_inds = [[ind]*len(part_names) ]

np.unique(all_types)
['A0-vibraphone', 'A1', 'A2-piano', 'A4', 'Alto Sax Solo', 'Alto Sax solo', 'B0', 'B1', 'B2', 'B4', 'Banjo Solo', 'Bari Sax Solo', 'Bari Sax Solo, Ensemble', 'Bass', 'Bass Solo', 'Bass Solo / Trombone Solo / Ensemble', 'Break (Half Time)', 'Bridge', 'C0', 'C1', 'C3', 'C4', 'Chorus 10', 'Chorus 11', 'Chorus 11,12', 'Chorus 7 / AA', 'Chorus 7 / BA', 'Chorus 8', 'Chorus 9', 'Chorus 9,10', 'Clarinet & Trombone Solo', 'Clarinet Solo', 'Clarinet Solo, Sax Solo', 'Clarinet Solo, Trombone Solo', 'Clarinet solo', 'Coda', 'Coda (Half Time)', 'Collective Improvisation', 'Cornet Solo', 'Cornet Solo, Ensemble', 'Drums', 'Drums Intro', 'Drums Solo', 'Ensemble', 'Ensemble / Piano Solo', 'Ensemble, Bari Sax Solo, Piano Solo', 'Ensemble, Bass Solo', 'Ensemble, Clarinet Solo', 'Ensemble, Guitar Solo', 'Ensemble, Percussion Solo', 'Ensemble, Scat Solo', 'Ensemble, Scat Solo, Head', 'Ensemble, Vocals Female', 'Ensemble,Sax Solo', 'Extra beats at the end of bass solo', 'Extra beats in drum solo', 'Extra measure!', 'Groove', 'Guitar Solo', 'Guitar Solo, Ensemble', 'Head', 'Interlude', 'Intro', 'Outro', 'Piano', 'Piano Solo', 'Piano Solo , Ensemble', 'Piano Solo, Bass Solo', 'Piano Solo, Bass Solo, Drums Solo', 'Piano Solo, Clarinet Solo', 'Piano Solo, Ensemble', 'Piano Solo, Head', 'Piano Solo, Horn Solo', 'Piano Solo, Vocals Female', 'Piano solo', 'Pickup', 'Sax Solo', 'Sax Solo, Piano Solo, Ensemble', 'Sax Solo, Trumpet Solo', 'Sax Solo, Vocals Female', 'Sax solo', 'Scat', 'Scat Solo', 'Scat, Banjo Solo', 'Scat, Clarinet Solo', 'Scatt Male, Scatt Female', 'Soprano Sax Solo', 'Tenor Sax Solo', 'Tenor Sax Solo, Bari Sax Solo', 'Tenor Sax Solo, Head', 'Tenor Sax Solo, Piano Solo', 'Tenor Sax Solo, Tenor Sax Solo, Drums Solo, Tenor Sax Solo', 'Tenor Sax Solo, Trombone solo', 'Tenor Sax Solo, Trumpet solo', 'Tenor Solo, Trumpet Solo, Ensemble', 'Tenor solo', 'Theme', 'Trade-off', 'Trade-off 8-8', 'Tradeoff', 'Trombone Solo', 'Trombone Solo, Collective Improvisation', 'Trombone Solo, Trumpet Solo', 'Trumbone Solo', 'Trumpet', 'Trumpet Solo', 'Trumpet Solo, Clarinet Solo, Ensemble', 'Trumpet Solo, Drums Solo', 'Trumpet Solo, Ensemble', 'Trumpet Solo, Head', 'Trumpet Solo, Piano Solo', 'Trumpet Solo, Piano Solo, Ensemble', 'Trumpet Solo, Tenor Sax Solo', 'Trumpet solo', 'Trumpet solo, Ensemble', 'Vamp', 'Vibraphone Solo', 'Vibraphone Solo, Guitar Solo', 'Vocals', 'Vocals Acapella', 'Vocals Female', 'Vocals Female , Trumpet Solo', 'Vocals Female, Ensemble', 'Vocals Male', 'Vocals Male, Vocals Female', 'Vocals female', 'Vocals, Collective Improvisation', 'pickup']
np.unique(all_forms)
['16-bar Blues', 'A', 'AA', "AA'", 'AA(*) (Half Time)', "AAB'A", 'AABA', 'AABA (Double Time Feel + 2 extra beats)', 'AABA (Double Time Feel)', 'AABA(*)', 'AABA(*) (Double Time)', 'AABA(*) (Faster Tempo)', 'AABA*', 'AABA*2', 'AABACD', 'AABBACD', 'AB', 'AB*2', 'AB*3', 'ABA', 'ABAC', 'ABAC(*)', 'ABAC*', 'ABB', 'ABCA', 'ABCD', 'AC', 'B', 'BA', 'BA, Coda', 'BAC', 'BB', 'Blues', 'Blues (10-bars)', 'Blues (14-bar)', 'Blues Bird', 'Blues Bird*2', 'Blues Bird*3', 'Blues Major', 'Blues Major*16', 'Blues Major*2', 'Blues Major*7', 'Blues Minor', 'Blues Minor*2', 'Blues*16', 'Blues*2', 'Blues*2 (extra 8-bars)', 'Blues*3', 'Blues*4', 'Blues*5', 'Blues*7', 'Bridge', 'C', 'CA', "CC'", 'CCD', 'CD', 'Coda', 'DDD', 'Interlude', 'Intro', 'Outro', 'Trumpet Solo', 'Vamp', 'Vamp*2', 'Vamp*3']


# Ideas:
# - list of segment types
# 	- instrument
# 	- function
# 	- confusion matrix between them
# 	- form labels
# 	- number of beats per segment
# - transition matrix between segment types




parameters = {'input_song': audio_path, 'input_times': "nofile", 'output_file': "analysis_of_{0}.txt".format(jaah_info.loc[id]["stem"])}
parameters = {'input_song': audio_path, 'input_times': "nofile", 'beat_times': all_onsets, 'downbeat_indices':[i for i in range(len(all_onsets)-1) if i%4==0], 'output_file': "analysis_of_{0}_guided.txt".format(jaah_info.loc[id]["stem"])}
segmenter.main(parameters)




plt.clf(),plt.subplot(1,3,1),plt.imshow(A_rep),plt.subplot(1,3,2),plt.imshow(A_loc),plt.subplot(1,3,3),plt.imshow(A_combined)





importlib.reload(segmenter)

# Do next:
then you can start running tests and evaluations:
- run algo on song
- evaluate result
- visualize result

# missing tally is only 1!
	
	

   <span class="s2">"path"</span><span class="o">:</span> <span class="s2">"$JAZZ_HARMONY_DATA_ROOT//Jazz_ The Smithsonian Anthology/CD1/Disc 1 - 20 - Swing That Music.flac"</span><span class="p">,</span> 



set(json_stems) == set(html_stems)

1. annotation json files: ("~/Documents/repositories/JAAH/annotations/*.json")
2. html files that list the DISC paths: ("~/Documents/repositories/JAAH/docs/data/*.html")
Read everything in list 2, make two lists: [filename, DISC PATH name]
From these, make PD dataframe of index (arbitrary number), audio filename, annotations json name, and some simple stats about the files:
	- number of parts
	- number of solos
	- instruments

How am I going to evaluate a single output?
I need the main solo junctions
I need 


# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
# sys.path.append(os.path.expanduser("~/Documents/repositories/JAAH"))
sys.path.append(os.path.expanduser("~/Dropbox/Ircam/code"))

jaah_ann_dir = os.path.expanduser("~/Documents/repositories/JAAH/annotations/maple_leaf_rag(hyman).json")
with open(jaah_ann_dir, 'rb') as f:
	j = json.load(f)

table = load_salami_table()
jazz = table.loc[table.CLASS=='jazz']
# salid = jazz.iloc[10]["SONG_ID"]
salid = 434
song = load_salami_song(salid, 1, 'uppercase')
# y, sr = librosa.load(get_salami_audio_filename(salid), sr=44100, mono=True)

import segmenter
# parameters = {'input_song': get_salami_audio_filename(salid), 'input_times': "/Users/jordan/Dropbox/AIST/demix/downbeats_salami_2.txt", 'output_file': "tmp_out.lab"}
parameters = {'input_song': "/Users/jordan/Documents/data/JAAH/Jazz_the_smithsonian_anthology_part1/CD1/Disc 1 - 01 - Maple Leaf Rag.flac", 'input_times': "nofile", 'output_file': "analysis_of_01-01.lab"}
segmenter.main(parameters)

