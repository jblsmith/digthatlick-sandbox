import os
import pandas as pd
import numpy as np
import scipy as sp
import librosa
import sys
import json

import glob
from bs4 import BeautifulSoup
import re

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


id, stem (for json and html), audio_path, 

targets = {}
for fi,fn in enumerate(html_files):
	with open(fn) as fp:
		soup = BeautifulSoup(fp)
	spans = soup.find_all('span')
	target = [span.text for span in spans if "JAZZ_HARMONY_DATA_ROOT" in span.text]
	targets[html_stems[fi]] = target

for k in targets.keys():
	[targets[k], ]
jaah_info = pd.DataFrame(data=html_stems)


clean_targets = [target.replace("$JAZZ_HARMONY_DATA_ROOT/",os.path.expanduser("~/Documents/data/JAAH/"))[1:-1] for target in targets]
jaah_info['clean_target'] = clean_targets

missing_tally = 0
found_tally = 0
for target in clean_targets:
	if not os.path.exists(target):
		missing_tally += 1
		print("Didn't find {0}".format(target))
	else:
		found_tally += 1

print(missing_tally)
print(found_tally)

np.array((html_stems, html_files, clean_targets))

# Do next:
- create linked list in the form of a dataframe with information
save the dataframe and git-save it
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

