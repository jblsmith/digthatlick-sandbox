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
	duration = ann_json['duration']
	# all_onsets = list(itertools.chain.from_iterable(all_onsets))
	return ann_json, part_names, part_types, part_forms, part_onsets, all_onsets, duration

def parse_all_annotations():
	all_names = []
	all_types = []
	all_forms = []
	all_onsets = []
	all_offsets = []
	all_inds = []
	part_lens = []
	jaah_info = pd.read_csv("audio_paths_index.csv")
	for ind in jaah_info.index:
		try:
			ann_json, part_names, part_types, part_forms, part_onsets, all_ons, duration = parse_annotation(ind)
			all_names += [p.strip().lower() for p in part_names]
			all_types += [p.strip().lower() for sub_list in part_types for p in sub_list if p]
			all_forms += [p.strip().lower() for sub_list in part_forms for p in sub_list if p]
			all_onsets += part_onsets
			all_offsets += part_onsets[1:] + [float(duration)]
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
	df['offset'] = all_offsets
	df['n_beats'] = part_lens
	df['words'] = [[word.replace(",","") for word in title.split()] for title in all_names]
	df['instruments'] = [[w for w in word_list if w in instrument_list] for word_list in df['words']]
	df['inst_qualifiers'] = [[w for w in word_list if w in instrument_qualifiers] for word_list in df['words']]
	df['funcs'] = [[w for w in word_list if w in func_list] for word_list in df['words']]
	df['residual'] = [[w for w in word_list if w not in func_list+instrument_list+block_words+instrument_qualifiers] for word_list in df['words']]
	return df

def get_basic_feats(ind):
	jaah_info = pd.read_csv("audio_paths_index.csv")
	audio_path = jaah_info.audio_path[ind]
	audio, sr = librosa.load(audio_path, sr=22050, mono=True)
	cqt = librosa.core.cqt(audio)
	mfcc = librosa.feature.mfcc(audio)
	rhyt = librosa.feature.rhythm.tempogram(audio)
	tempo, beat_inds = librosa.beat.beat_track(audio,sr=sr)
	return cqt, mfcc, rhyt, tempo, beat_inds, audio, sr

# The input is a transition matrix where TM[i,j] indicates the probability of stepping from points i to j (or some value correlated to the probability).
# Enforcements:
# 	- must be upper triangular, with everything on or below the main diagonal equal to -inf, because these transitions are impossible.
# def decode_transition_matrix(tm):
# 	inf_mask = np.tril(np.ones_like(tm))
# 	inf_mask[inf_mask==1] = -np.inf
# 	assert np.all(np.tril(tm)==inf_mask)
# 	# best_next_steps[i,:] = [best_next_index, unit_cost_of_leaping_to_that_index]
# 	best_next_steps = np.zeros((tm.shape[0],2))
# 	best_next_steps[-1,:] = [0, 0]						# From END, leap to beginning at cost 0
# 	best_next_steps[-2,:] = [tm.shape[0]-1, tm[-2,-1]]	# From END-1, leap to END at cost defined by TM
# 	for ti in range(tm.shape[0]-3, -1, -1):
# 		# for ti in range(tm.shape[0]-3, tm.shape[0]-6, -1):
# 		tj_opts = np.where(~np.isinf(tm[ti,:]))[0]
# 		tj_costs = best_next_steps[tj_opts,1] + tm[ti,tj_opts]
# 		choice = np.argmin(tj_costs)
# 		best_next_steps[ti,0] = tj_opts[choice]
# 		best_next_steps[ti,1] = tj_costs[choice]
# 	# Now, backtrace from start!
# 	path = [int(best_next_steps[0,0])]
# 	while (tm.shape[0]-1) not in path:
# 		path += [int(best_next_steps[path[-1],0])]
# 		# If we get caught in a loop somehow, this assertion should catch it.
# 		assert len(path) == len(np.unique(path))
# 	return best_next_steps, [0] + path

def decode_transition_matrix(tm_in):
	tm = tm_in.copy()
	bmat = basic_matrix(tm)[:-1,:-1]
	tm[np.isnan(bmat)] = np.nan
	# best_next_steps[i,:] = [best_next_index, unit_cost_of_leaping_to_that_index]
	best_next_steps = np.zeros((tm.shape[0],2))
	best_next_steps[-1,:] = [0, 0]						# From END, leap to beginning at cost 0
	best_next_steps[-2,:] = [tm.shape[0]-1, tm[-2,-1]]	# From END-1, leap to END at cost defined by TM
	for ti in range(tm.shape[0]-3, -1, -1):
		# for ti in range(tm.shape[0]-3, tm.shape[0]-6, -1):
		tj_opts = np.where(np.isfinite(tm[ti,:]))[0]
		tj_costs = best_next_steps[tj_opts,1] + tm[ti,tj_opts]
		choice = np.nanargmin(tj_costs)
		best_next_steps[ti,0] = tj_opts[choice]
		best_next_steps[ti,1] = tj_costs[choice]
	# Now, backtrace from start!
	path = [int(best_next_steps[0,0])]
	while (tm.shape[0]-1) not in path:
		path += [int(best_next_steps[path[-1],0])]
		# If we get caught in a loop somehow, this assertion should catch it.
		assert len(path) == len(np.unique(path))
	return best_next_steps, [0] + path

# Original script for this... but did I think through all the edge cases? Rewriting above to assert more properties of the transition matrix.
# def decode_transition_matrix(tm):
# 	best_next_steps = np.zeros((tm.shape[0],2))
# 	best_next_steps[-1,:] = [0, tm[-1,-1]]
# 	best_next_steps[-2,:] = [tm.shape[0]-1, tm[-2,-1]]
# 	for ti in range(tm.shape[0]-3, -1, -1):
# 		tj_opts = np.where(tm[ti,:]>0)[0]
# 		tj_costs = best_next_steps[tj_opts,1] + tm[ti,tj_opts]
# 		choice = np.argmin(tj_costs)
# 		best_next_steps[ti,0] = tj_opts[choice]
# 		best_next_steps[ti,1] = tj_costs[choice]
# 	# Now, backtrace from start!
# 	path = [int(best_next_steps[0,0])]
# 	while (tm.shape[0]-1) not in path:
# 		path += [int(best_next_steps[path[-1],0])]
# 		# If we get caught in a loop somehow, this assertion should catch it.
# 		assert len(path) == len(np.unique(path))
# 	return best_next_steps, path

# This is maybe a silly idea, but it's a transition matrix where steps on different planes are permitted.
# We solve it just by taking the min across different planes, and including the argmin when we backtrace.
def decode_3d_transition_matrix(tm):
	tm_min_costs = np.nanmin(tm,axis=0)
	tm_arc_choices = np.nanargmin(tm,axis=0)
	best_next_steps, path = decode_transition_matrix(tm_min_costs)
	path = [0] + path
	path_z = [tm_arc_choices[i,j] for (i,j) in zip(path[:-1],path[1:])]
	return best_next_steps, np.array((path[1:], path_z)).transpose()

def richer_agg(feat, inds, aggfuncs):
	feat_versions = [librosa.util.sync(feat, inds, aggregate=aggfunc, pad=True, axis=-1) for aggfunc in aggfuncs]
	feat_out = np.concatenate(feat_versions,axis=0)
	return feat_out

def convert_to_percentile(mat):
	# Create a new matrix where mat[i,j] = ordinal_rank[mat[i,j]] within values of mat
	# BUT, also want to remove all zero entries from consideration. And then we want to normalize, so values range from 0 to 1.
	assert np.min(mat)>=0
	# mat = sp.random.rand(4,4)
	# mat = np.triu(mat,1)
	arr_row = np.reshape(mat,(np.prod(mat.shape)))
	arr_ord = np.argsort(arr_row,axis=0)
	# arr_ord contains the sort order of the original elements.
	# Next: force relevant N items into range [1/N, 1]
	first_non_zero = np.where(arr_row[arr_ord]>0)[0][0]
	ord_vals = np.zeros_like(arr_ord).astype(float)
	ord_vals[first_non_zero:] = np.linspace(0,1,len(arr_ord)-first_non_zero+1)[1:]
	new_row = arr_row*0
	new_row[arr_ord] = ord_vals
	# Just to convince ourselves that this is how to reshape the vector into the matrix:
	assert np.all(mat == np.reshape(arr_row,mat.shape))
	mat_ord = np.reshape(new_row, mat.shape)
	return mat_ord

def compute_gaussian_krnl(M):
	from scipy import signal
	"""Creates a gaussian kernel following Foote's paper."""
	g = signal.gaussian(M, M // 3., sym=True)
	G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
	G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
	G[:M // 2, M // 2:] = -G[:M // 2, M // 2:]
	return G

def basic_matrix(mat):
	assert mat.shape[0] == mat.shape[1]
	length = mat.shape[0]+1
	cost_mat = np.ones((length,length))
	cost_mat = np.tril(cost_mat)
	cost_mat[cost_mat==1] = np.nan
	cost_mat[cost_mat==0] = np.inf
	return cost_mat

def base_cost_matrix(mat):
	## Base cost
	#  There is a base cost to traversing a length L: it's L. That's the ensure that every leap is technically allowed.
	#  Thereafter, the real "costs" will be negative values added to this basis.
	base_cost = basic_matrix(mat)
	for i in range(base_cost.shape[0]):
		for j in range(i+1,base_cost.shape[1]):
			base_cost[i,j] = j-i
	return base_cost

def leap_cost_matrix(mat, max_delta=10):
	## Leap cost
	#  We modify the cost of a leap of length L depending on whether L is near to some power of 2.
	leap_cost = basic_matrix(mat)
	powers_of_two = np.array([np.power(2,n) for n in range(10)])
	for i in range(leap_cost.shape[0]):
		for j in range(i+1,leap_cost.shape[1]):
			delta = j-i
			if delta<=max_delta:
				power_cost = np.min(np.abs(powers_of_two-delta))
				leap_cost[i,j] = (power_cost+1)
	return leap_cost

def block_novelty_matrix(mat, gaussness=0.5, max_delta=10):
	## Block novelty reward
	#  Looks at novelty of [i,i+d] with respect to [i-d,i]
	#  Scales the region using a checkerboard kernel that is gaussian (1) or not (0) or half-gaussian, half-flat (0.5, default)
	cost_mat = basic_matrix(mat)
	for i in range(cost_mat.shape[0]-1):
		for j in range(i+1,cost_mat.shape[1]):
			delta = j-i
			if (delta<=max_delta) & (i-delta>=0) & (delta>=1):
				checkerboard = mat[i-delta:j,i-delta:j]
				check_kernel_gauss = compute_gaussian_krnl(delta*2)
				check_kernel_flat = np.block([[np.ones((delta,delta)), -np.ones((delta,delta))], [-np.ones((delta,delta)), np.ones((delta,delta))]])
				check_kernel = gaussness*check_kernel_gauss + (1-gaussness)+check_kernel_flat
				cost_mat[i,j] = np.mean(check_kernel * checkerboard)
	return cost_mat

def repetition_matrix(mat, kind, map_to_range=True, max_delta=10):
	## Repetition reward
	#  Looks at stripes or blocks [k,k+d] for k in [0, 1, ..., i-d].
	#  You must specify the kind: 'stripe' or 'block'.
	#  TODO: should I be ignoring the diagonals of these blocks? Same with novelty matrix blocks above.
	#  The minimum cost of these stripes is the cost of the arc, and we also record the k to know where the repetition started.
	assert kind in ['block','stripe']
	cost_mat = basic_matrix(mat)
	cost_mat_pre_start = basic_matrix(mat)
	for i in range(cost_mat.shape[0]-1):
		for j in range(i+1,cost_mat.shape[1]):
			delta = j-i
			if (delta<=max_delta) & (i-delta>=0) & (delta>=1):
				if kind=='block':
					# main_block = mat[i:j,i:j]
					pre_rep_options = [mat[k:k+delta, i:j] for k in range(0,i-delta+1)]
				elif kind == 'stripe':
					# main_block = mat[range(i,j),range(i,j)]
					pre_rep_options = [mat[range(k,k+delta), range(i,j)] for k in range(0,i-delta+1)]
				pre_rep_opt_costs = [np.mean(tmp_mat) for tmp_mat in pre_rep_options]
				# rep_weight = np.max(pre_rep_opt_costs) - np.min(pre_rep_opt_costs)
				rep_weight = 1.0
				cost_mat[i,j] = np.min(pre_rep_opt_costs) * rep_weight
				cost_mat_pre_start[i,j] = np.argmin(pre_rep_opt_costs)
	if map_to_range:
		minval = np.nanmin(cost_mat[np.isfinite(cost_mat)])
		maxval = np.nanmax(cost_mat[np.isfinite(cost_mat)])
		rangeval = maxval - minval
		assert maxval > 0
		assert rangeval > 0
		cost_mat = cost_mat - maxval
		cost_mat = cost_mat / rangeval
		assert np.all(cost_mat[np.isfinite(cost_mat)]<=0)
	return cost_mat, cost_mat_pre_start

# def block_repetition_matrix(mat, map_to_range=True):
# 	## Block repetition reward
# 	#  Looks at blocks [k,k+d] for k in [0, 1, ..., i-d].
# 	#  TODO: should I be ignoring the diagonals of these blocks? Same with novelty matrix blocks above.
# 	#  The minimum cost of these stripes is the cost of the arc, and we also record the k to know where the repetition started.
# 	cost_mat_blockrep = basic_matrix(mat)
# 	cost_mat_pre_start = basic_matrix(mat)
# 	for i in range(cost_mat_blockrep.shape[0]-1):
# 		for j in range(i+1,cost_mat_blockrep.sahpe[1]):
# 			delta = j-i
# 			if (i-delta>=0) & (delta>=1):
# 				main_block = mat[i:j,i:j]
# 				pre_rep_options = [mat[k:k+delta, i:j] for k in range(0,i-delta+1)]
# 				pre_rep_opt_costs = [np.mean(tmp_mat) for tmp_mat in pre_rep_options]
# 				# rep_weight = np.max(pre_rep_opt_costs) - np.min(pre_rep_opt_costs)
# 				rep_weight = 1.0
# 				cost_mat_blockrep[i,j] = np.min(pre_rep_opt_costs) * rep_weight
# 				cost_mat_pre_start[i,j] = np.argmin(pre_rep_opt_costs)
# 	if map_to_range:
# 		minval = np.min(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)])
# 		maxval = np.max(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)])
# 		rangeval = maxval - minval
# 		assert maxval > 0
# 		assert rangeval > 0
# 		cost_mat_blockrep = cost_mat_blockrep - maxval
# 		cost_mat_blockrep = cost_mat_blockrep / rangeval
# 		assert np.all(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)]<=0)
# 	return cost_mat_blockrep, cost_mat_pre_start

# TODO: how will I scale these cost matrices?
# - scaling with respect to each other
# - scaling the values within them

# def stripe_repetition_matrix(mat, map_to_range=True):
# 	## Stripe repetition reward
# 	#  Looks at stripes [k,k+d] for k in [0, 1, ..., i-d].
# 	#  The minimum cost of these stripes is the cost of the arc, and we also record the k to know where the repetition started.
# 	assert mat.shape[0] == mat.shape[1]
# 	length = mat.shape[0]+1
# 	cost_mat_blockrep = np.zeros((length,length)) - np.inf
# 	cost_mat_pre_start = np.zeros((length,length)) - np.inf
# 	for i in range(length-1):
# 		for j in range(i+1,length):
# 			delta = j-i
# 			if (i-delta>=0) & (delta>=1):
# 				main_block = mat[range(i,j),range(i,j)]
# 				pre_rep_options = [mat[range(k,k+delta), range(i,j)] for k in range(0,i-delta+1)]
# 				pre_rep_opt_costs = [np.mean(tmp_mat) for tmp_mat in pre_rep_options]
# 				# rep_weight = np.max(pre_rep_opt_costs) - np.min(pre_rep_opt_costs)
# 				rep_weight = 1.0
# 				cost_mat_blockrep[i,j] = np.min(pre_rep_opt_costs) * rep_weight
# 				cost_mat_pre_start[i,j] = np.argmin(pre_rep_opt_costs)
# 	if map_to_range:
# 		minval = np.min(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)])
# 		maxval = np.max(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)])
# 		rangeval = maxval - minval
# 		assert maxval > 0
# 		assert rangeval > 0
# 		cost_mat_blockrep = cost_mat_blockrep - maxval
# 		cost_mat_blockrep = cost_mat_blockrep / rangeval
# 		assert np.all(cost_mat_blockrep[~np.isinf(cost_mat_blockrep)]<=0)
# 	return cost_mat_blockrep, cost_mat_pre_start

def plot_slices(mat3d, fig=None, filename=None):
	plt.clf
	n = mat3d.shape[2]
	y = int(np.round(np.sqrt(n)))
	x = int(np.ceil(n/y))
	if fig is not None:
		plt.figure(fig)
	for i in range(n):
		plt.subplot(y,x,i+1)
		plt.imshow(mat3d[:,:,i])
	if filename is None:
		plt.savefig("tmp.pdf")
	else:
		plt.savefig(filename)

def setup_toy_mat(xsize=16, repgroups = [ (4,[2,6,12]) ], blockgroups=[ (2,[0,10]), (2,[2,6,12]) ]):
	mat = np.ones((xsize,xsize))
	mat -= np.eye(xsize)
	if repgroups is not None:
		assert [xsize > grpinfo[0] + np.max(grpinfo[1]) for grpinfo in repgroups]
		for grpinfo in repgroups:
			grp_len, grp_starts = grpinfo
			for i,j in itertools.combinations(grp_starts,2):
				mat[np.arange(grp_len)+i,np.arange(grp_len)+j] = 0
	if blockgroups is not None:
		assert [xsize > grpinfo[0] + np.max(grpinfo[1]) for grpinfo in blockgroups]
		for grpinfo in blockgroups:
			grp_len, grp_starts = grpinfo
			for i,j in itertools.combinations_with_replacement(grp_starts,2):
				mat[i:i+grp_len,j:j+grp_len] = 0
	mat = np.min((mat,mat.transpose()),axis=0)
	return mat

def cqt_to_chroma_bass_treble(cqt, split=40):
	bass_cqt = cqt[:split,:].copy()
	treb_cqt = cqt[split:,:].copy()
	chroma = np.zeros((12,cqt.shape[1]))
	for i in range(int(cqt.shape[0]/12)):
		chroma += np.abs(cqt[i*12:(i+1)*12,:])
	return chroma/12, bass_cqt, treb_cqt

