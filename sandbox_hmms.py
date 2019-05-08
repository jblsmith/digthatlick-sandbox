def test_cost_functions():
mat = setup_toy_mat(6,[(2,[0,4])],None)
mat = setup_toy_mat(10,None,[(2,[0,4]), (2,[2,8])])

mat_pad = np.block([[mat, np.zeros((mat.shape[0],1))], [np.zeros((1,mat.shape[0])), 0]])
cost_base = leap_cost_matrix(mat) * (base_cost_matrix(mat)+1)
cost_bnov = block_novelty_matrix(mat, gaussness=0.5)
cost_bnov[0,1] = np.max(cost_bnov[np.isfinite(cost_bnov)])
cost_brep, brep_starts = repetition_matrix(mat,'block')
cost_brep[0,1] = np.max(cost_brep[np.isfinite(cost_brep)])
cost_srep, srep_starts = repetition_matrix(mat,'stripe')
cost_srep[0,1] = np.max(cost_srep[np.isfinite(cost_srep)])
plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_bnov, cost_brep, cost_srep),axis=2),None,"tmp1.pdf")
plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_base*cost_bnov, cost_base*cost_brep, cost_base*cost_srep),axis=2),None,"tmp2.pdf")

decode_transition_matrix(cost_base*cost_bnov)[1]
decode_transition_matrix(cost_base*cost_brep)[1]
decode_transition_matrix(cost_base*cost_srep)[1]


# bnov_i, brep_i, srep_i = [np.isfinite(cm) for cm in [cost_bnov, cost_brep, cost_srep]]
cost_mat = cost_base.copy()
for cm in [cost_bnov, cost_brep, cost_srep]:
	inds = np.isfinite(cm)
	cost_mat[inds] += cm[inds] * cost_base[inds]



# Method 1: sum of 


decode_transition_matrix(cost_mat)[1]
plt.clf(), plt.imshow(cost_mat), plt.savefig("tmp2.pdf")
decode_transition_matrix(np.nanmin((cost_bnov, cost_base),axis=0))[1]

plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_bnov, cost_brep, cost_srep, np.nansum((cost_srep,cost_brep),axis=0)),axis=2))


# TODO: change all -infs to np.nans, so that I can easily np.nansum, nanprod, nanmax, etc. all the cost matrices?
# 		but I will really need to write out some equations of the similarity costs to figure out what I want, how to combine things.
# TODO: consider two-hop costs? hop bigrams, in other words? So you have to consider the hop (i,j,k) followed by (j,k,l), etc.
# 		second thought... nah.

decode_transition_matrix(cost_base)[1]
decode_transition_matrix(cost_base)[1]
decode_transition_matrix(cost_base)[1]



mat = setup_toy_mat(10,None,[(2,[0,4]), (2,[2,8])])
mat_pad = np.block([[mat, np.zeros((mat.shape[0],1))], [np.zeros((1,mat.shape[0])), 0]])
# That should have form ABACB, each 2 time units long.
cost_base = leap_cost_matrix(mat)
cost_bnov = block_novelty_matrix(mat, gaussness=0.5)
cost_brep, brep_starts = block_repetition_matrix(mat)
cost_srep, srep_starts = stripe_repetition_matrix(mat)
# The final cost matrix should have:
# - where cost_bnov is undefined, cost_base
# - where cost_bnov is defined, cost_bnov * cost_base
cm = np.zeros_like(cost_base) -np.inf
base_inds = ~np.isinf(cost_base)
bnov_inds = ~np.isinf(cost_bnov)
cm[base_inds] = cost_base[base_inds]
cm1 = cm.copy()
cm2 = cm.copy()
cm1[bnov_inds] *= cost_bnov[bnov_inds]
cm2[bnov_inds] += cost_brep[bnov_inds]
info, path = decode_transition_matrix(cm1)
info, path = decode_transition_matrix(cm2)
info, path = decode_3d_transition_matrix(np.stack((cm,cm1,cm2),axis=2))
plt.clf(),plt.imshow(cost_bnov)
plt.savefig("tmp2.pdf")
plt.
cost_bnov_orig = cost_bnov.copy()
cost_bnov[np.isinf(cost_bnov)] = np.max(cost_base)
cost_bnov += np.tril(cost_base)
info, path = decode_transition_matrix(cost_base * cost_bnov)
info, path = decode_transition_matrix(cost_base)
info, path = decode_transition_matrix(cost_bnov)
plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_bnov, cost_brep),axis=2))

mat = setup_toy_mat(8,[(4,[0,4])], None)

mat = setup_toy_mat(34,
	repgroups = [(4,[2, 6, 18, 22])],
	blockgroups = [(2,[0,10]), (4,[14,28])])

cost_base = base_cost_matrix(mat)
cost_bnov = block_novelty_matrix(mat, gaussness=0.5)
cost_bseq, starts_bseq = block_repetition_matrix(mat)
cost_bnov[np.isinf(cost_bnov)] = 0
cost_bseq[np.isinf(cost_bseq)] = 0
plot_slices(np.stack((mat, cost_bnov, cost_bseq, cost_base+cost_bnov+cost_bseq),axis=2))


info, path = decode_transition_matrix(cost_base + cost_bnov)


plt.clf()
plt.imshow(cm)
plt.savefig("tmp.pdf")


np.max((cost_base, cost_bnov),axis=0)


cqt, mfcc, rhyt, tempo, beat_inds, audio, sr = get_basic_feats(5)
chroma, bcqt, tcqt = cqt_to_chroma_bass_treble(cqt)
b_ccqt = richer_agg(np.real(np.abs(chroma)), beat_inds, [np.mean, np.max, np.std]).transpose()
b_bcqt = richer_agg(np.real(np.abs(bcqt)), beat_inds, [np.mean, np.max, np.std]).transpose()
b_tcqt = richer_agg(np.real(np.abs(tcqt)), beat_inds, [np.mean, np.max, np.std]).transpose()
b_mfc = richer_agg(np.real(mfcc), beat_inds, [np.mean, np.max, np.std]).transpose()
b_rhy = richer_agg(np.real(rhyt), beat_inds, [np.mean, np.max, np.std]).transpose()
test_len = 500

# ssm1 = sp.spatial.distance.cdist(b_cqt[:test_len], b_cqt[:test_len], metric='cosine')
# ssm2 = sp.spatial.distance.cdist(b_mfc[:test_len], b_mfc[:test_len], metric='cosine')
# ssm3 = sp.spatial.distance.cdist(b_rhy[:test_len], b_rhy[:test_len], metric='cosine')
# ssms = np.array([convert_to_percentile(mat) for mat in [ssm1, ssm2, ssm3]]).transpose((1,2,0))
ssms = [sp.spatial.distance.cdist(b_mat[:test_len], b_mat[:test_len], metric='cosine') for b_mat in [b_ccqt, b_bcqt, b_tcqt, b_mfc, b_rhy, ]]
plot_slices(np.array(ssms).transpose((1,2,0)),1,"ssms-reg.pdf")
ssms = np.array([convert_to_percentile(mat) for mat in ssms]).transpose((1,2,0))
plot_slices(ssms,1,"ssms-perc.pdf")

# Compute cost matrices
mat = ssms[:,:,0].copy()
mat_pad = np.block([[mat, np.zeros((mat.shape[0],1))], [np.zeros((1,mat.shape[0])), 0]])
cost_base = leap_cost_matrix(mat) * (base_cost_matrix(mat)+1)
cost_bnov = block_novelty_matrix(mat, gaussness=0.5)
cost_bnov[0,1] = np.max(cost_bnov[np.isfinite(cost_bnov)])
cost_brep, brep_starts = repetition_matrix(mat,'block')
cost_brep[0,1] = np.max(cost_brep[np.isfinite(cost_brep)])
cost_srep, srep_starts = repetition_matrix(mat,'stripe')
cost_srep[0,1] = np.max(cost_srep[np.isfinite(cost_srep)])
plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_bnov, cost_brep, cost_srep),axis=2),None,"tmp1.pdf")
plt.clf(),plot_slices(np.stack((mat_pad, cost_base, cost_base*cost_bnov, cost_base*cost_brep, cost_base*cost_srep),axis=2),None,"tmp2.pdf")

decode_transition_matrix(cost_base*cost_srep)[1]


cost_base = base_cost_matrix(mat)
cost_bnov = block_novelty_matrix(mat, gaussness=0.5)
cost_bseq, starts_bseq = block_repetition_matrix(mat)
plot_slices(np.stack((cost_base, cost_bnov, cost_bseq),axis=2))

# Decode transitions by weighting two matrices
cm = cost_base.copy() + 1
cm[~np.isinf(cost_bnov)] += 12*cost_bnov[~np.isinf(cost_bnov)]
info, path = decode_transition_matrix(cm)
justifications = [starts_bseq[i,j] for [i,j] in zip(path[:-1],path[1:])]
for x in zip(path[:-1],path[1:], justifications):
	print(x)

def make_stripe_cost_matrix(mat):
	# The output should have pixel (i,j) represent the minimum stripe similarity to some previous segment of length j-i.
	# First, we make a matrix showing the average value of all diagonal stripes.
mat = ssm1.copy()
mat = np.ones((15,15))
mat -= np.eye(15)
mat[range(4),range(4,8)] = 0
mat[range(4),range(11,15)] = 0
mat[range(4,8),range(11,15)] = 0
biz_mat = np.fliplr(np.transpose(np.fliplr(mat)))
biz_mattm = librosa.segment.recurrence_to_lag((np.triu(biz_mat)), pad=False, axis=0)
mattm = np.fliplr(np.transpose(np.fliplr(biz_mattm)))
plt.imshow(mattm)
plt.savefig("tmp2.pdf")

# block costs

# This tells us how to leap from i to j when (j-i)>=0. But what about the first leap? We cannot make the step (0,4) in this scheme.
# So, we must set up a set of initial options.
# Or, we could just allow i:i+1 to always be allowed, with some steep cost equal to the length spanned. The length spanned can be our maximum cost for each span.



# Block costs
# new_block[t,l] = how well mat[t:t+l] stands out against background
# rep_block[t,l] = how well mat[t:t+l] indicates some previous repetition at k, or mat[k:k+l]
mat = ssm1.copy()
length = mat.shape[0]
block_sizes = [1,2,4,8]
rep_block = np.zeros((length,length,len(block_sizes)))
for k,bs in enumerate(block_sizes):
	for i in range(bs,length-1):
		for j in range(i,length):
			rep_block[i,j,k] = np.mean(mat[i:i+bs,j:j+bs])

for k,bs in enumerate(block_sizes):
	plt.subplot(2,2,k+1)
	plt.imshow(rep_block[:,:,k])

plt.savefig("tmp.pdf")

for i in range(length):
	for j in range(i,length):
		
for t in range(length):
	for l in range(length-t): #since the time lag cannot exceed the remaining length of the song
		rep_block[t,l] = mat[t:t+l,t:t+l]
		
	



new_block[i,j] = how well mat[i:j,i:j] stands out against background
background could be [i-j:i], or something else?
first, get a matrix where newmat[i,j] shows the average of the block
then, block operations should be simple (although there cannot be a kernel on top of it)

rep_block[i,j] = how well mat[i:j,i:j] indicates some previous repetition
there should be with this the index k where mat[k,k+j-i] is most similar to mat[i,j]



# Pseudo equation:
length = mat.shape[0]
inf_mask = np.tril(np.ones_like(mat))
inf_mask[inf_mask==1] = np.inf
mat_with_shifts = np.zeros([length,length,length]) + np.inf
mat_with_shifts[:,:,0] = mat + inf_mask
for i in range(1,length):
	# matrix = previous answer, shifted, diagonally down and right, plus original snippet.
	# Both parts scaled so that each copy from 1 to N is weighted equally.
	# mat_with_shifts[i:,i:,i] = (mat_with_shifts[i:,i:,i-1]*i + mat[:-i,:-i]) / (i+1)
	mat_with_shifts[:-i,:-i,i] = (mat_with_shifts[:-i,:-i,i-1]*i + mat[i:,i:]) / (i+1)
	mat_with_shifts[:,:,i] += inf_mask
	# mat_with_shifts[i:,i:,i] 

plt.subplot(2,2,1)
plt.imshow(mat_with_shifts[:,:,0])
plt.subplot(2,2,2)
plt.imshow(mat_with_shifts[:,:,1])
plt.subplot(2,2,3)
plt.imshow(mat_with_shifts[:,:,3])
plt.subplot(2,2,4)
plt.imshow(mat_with_shifts[:,:,4])
plt.savefig("tmp.pdf")

rep_cost[i,j] = np.min (and argmin) [length (j-i) diagonal stripes starting after 0 and before i]
rep_cost = np.zeros_like(mat)
for i in range(mat.shape[0]):
	for j in range(mat.shape[0]):
		length = j-i
		if length > i


mat = librosa.segment.recurrence_to_lag(mat, pad=False, axis=0)
# Now, the repetitions (NW-SE diagonals) are in the columns (N-S)
new_mat = np.triu(np.ones_like(mat))
for i in range(new_mat.shape[0]):
	new_mat[i,i*2<np.arange(new_mat.shape[0])] = 0


new_mat[i,j] = 0 if i*2<j
for i in range(mat.shape[0]):
	for j in range(i,mat.shape[0]):
		new_mat[i,j] = 
plt.clf()
plt.imshow(mat)
plt.savefig("tmp.pdf")


	# This will be the average of the diagonal stripe


# Need to formalize this part.
# Also to-do: turn SSM into a transition matrix expressing block or segment costs.
# Want TM[i,j] to express block cost of calling [i,j] a block (must be compared to [i-j,i]?)
# Want TM[i,j] to express stripe cost of calling [i,j] a repeated sequence
# Want these to also express cost of calling them a NEW thing vs. an OLD thing?





dist_base_mat = np.zeros_like(ssms[0])
for i in range(dist_base_mat.shape[0]):
	dist_base_mat[i,i:] = np.arange(dist_base_mat.shape[0]-i)

min_step = 2
max_step = 5
ssm3d = np.stack([np.tril(np.triu(ssm + dist_base_mat,min_step),max_step) for ssm in ssms])
info, path3d = decode_3d_transition_matrix(ssm3d)
vertices = [0] + list(path3d[:,0])
edges = path3d[:-1,1]
plt.clf()
plt.subplot(1,2,1)
# plt.imshow(ssm3d[0])
plt.imshow(ssms[0])
plt.subplot(1,2,2)
# plt.imshow(ssm3d[1])
plt.imshow(ssms[1])
for i in range(len(edges)):
	plt.subplot(1,2,edges[i]+1)
	plt.plot([vertices[i+1],vertices[i+1]],[vertices[i],vertices[i]],'o-', color='white')

plt.savefig("tmp.pdf")

tm = np.zeros_like(ssm) + ssm
# tm *= dist_base_mat
tm += dist_base_mat
tm = np.tril(np.triu(tm,min_step),max_step)
model, path = decode_transition_matrix(tm)
path = [0] + path
plt.clf()
plt.imshow(ssm)
plt.plot(path[1:],path[:-1],'o-', color='black')
plt.savefig("tmp.pdf")

# Just use cqt for now.
# Set up transition matrix:
tm_size = 15
tm = np.ones([tm_size,tm_size])
# Basic contraints:
# from i, must transition to some future step between min and max steps away
min_step = 1
max_step = 4
tm = tm * np.tril(np.triu(np.ones_like(tm), min_step),max_step)
# normalize rows
# tm = (tm / np.tile(np.sum(tm,axis=1),(tm_size,1)).transpose())

# Find min cost from 0 to 14 with only steps allowed that have non-zero costs.
# Initial case: t=14

best_path_from = {i:{'step':0,'cost':np.inf} for i in range(tm_size)}
best_path_from[tm_size-1]['step'] = 0
best_path_from[tm_size-1]['cost'] = tm[tm_size-1,tm_size-1]
best_path_from[tm_size-2]['step'] = tm_size-1
best_path_from[tm_size-2]['cost'] = tm[tm_size-2,tm_size-1]
for ti in range(12,-1,-1):
	# for ti in range(13,0,-1):
	tj_opts = np.where(tm[ti,:]>0)[0]
	tj_options = [(tj,best_path_from[tj]['cost'] + tm[ti,tj]) for tj in tj_opts]
	tj_steps, tj_costs = zip(*tj_options)
	# These are the next step and the cost of the total path using that step, respectively, for all future options.
	choice = np.argmin(tj_costs)
	best_path_from[ti]['step'] = tj_steps[choice]
	best_path_from[ti]['cost'] = tj_costs[choice] #best_path_from[tj_steps[choice]]['cost'] + tj_costs[choice]

# Now, backtrace from start!
path = []
i = 0
cost = best_path_from[i]['cost']
next_step = best_path_from[i]['step']
while next_step>0:
	path += [next_step]
	next_step = best_path_from[next_step]['step']



	
	np.argmin(tj_opt_costs.values())
	cost_opts = {tj:}
	what are path options from ti onwards?
	for tj in path_options:
		cost_opts += [cost_ti_tj + cost_from_tj]
	keep the best cost_opt as being on the best_path_from[ti]
	for tj in range(14,ti,-1)
		
		

# What is the best path starting from ti?
for ti in range(13,0,-1):
	# Consider the future vertex tj:
	for tj in range(ti,14):
		cost_ti_tj = tm[ti,tj]
		cost_tj_end = best_path_from[tj]['cost']
		cost_ti_end
		
		# WhIs the path from ti-to-tj + the path from tj-end better than the path from ti-end?
		


sp = np.ones_like(tm)
librosa.sequence.viterbi(sp, tm)
librosa.sequence.transition_local(n_states=10,width=3)

tm = tm / np.tile(np.sum(tm,axis=1), (tm.shape[1],))
librosa.sequence.viterbi(prob=np.ones_like(tm), transition=tm)


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

