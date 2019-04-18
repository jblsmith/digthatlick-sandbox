import dataset
import os
import pandas as pd

def load_metadata_rows(table_field, melid=None, trackid=None):
	database_path = os.path.expanduser("~/Documents/data/WeimarJazzDatabase/wjazzd.db")
	assert table_field in ['beats', 'composition_info', 'db_info', 'esac_info', 'melid', 'melody', 'melody_type', 'notes', 'popsong_info', 'record_info', 'sections', 'solo_info', 'sqlite_sequence', 'track_info', 'transcription_info']
	assert not ((melid is None) & (trackid is None)) # Must specify either melid or trackid!
	with dataset.connect('sqlite:///' + database_path) as tx:
		if melid is not None:
			wanted_rows = tx[table_field].find(melid=melid)
		elif trackid is not None:
			wanted_rows = tx[table_field].find(trackid=trackid)
	rows = [row for row in wanted_rows]
	rows_as_df = pd.DataFrame(rows)
	return rows_as_df

def get_structure(melid, sec_type, add_solo_offset=False):
	assert sec_type in ['CHORD', 'CHORUS', 'FORM', 'IDEA', 'PHRASE']
	section_df = load_metadata_rows("sections", melid=melid)
	relevant_rows = section_df.loc[section_df.type==sec_type]
	# The "start" and "end" values in this table refer to the start and end NOTES in the melody table.
	melody_df = load_metadata_rows("melody", melid=melid)
	# Note: the times in THIS table are with respect to the SOLO, not the original recording.
	# If you want the times with respect to the original track, set add_solo_offset=True:
	if add_solo_offset:
		transcription_info = load_metadata_rows("transcription_info", melid)
		global_offset = transcription_info.loc[0]["solostart_sec"]
	else:
		global_offset = 0
	onset_times = [melody_df.iloc[x]["onset"] + global_offset for x in relevant_rows.start]
	offset_times = [melody_df.iloc[x]["onset"] + melody_df.iloc[x]["duration"] + global_offset for x in relevant_rows.end]
	new_cols = pd.DataFrame({'onset':onset_times, 'offset':offset_times}, index=relevant_rows.index)
	both_dfs = pd.concat([relevant_rows, new_cols],axis=1,sort=False)
	return both_dfs

get_structure(130, 'PHRASE', False)
