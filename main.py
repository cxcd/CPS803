import util
import generate
import torch
import numpy as np
import util
import dataprocess
import os
import pretty_midi
from operator import itemgetter

"""
# Loads only the pitches from preprocessed data
# If no parameters, load all the data
data = util.load_all_predata_pitchonly()

# Writes pitches to a midi file
util.write_piano_midi(util.pitches_to_midi(some_array))

"""

def get_processed_midi(file_num):
	return dataprocess.pretty_midi_to_event(util.here("twinkle.midi"))


def get_pitches(n):
	data = []
	# Get the data
	arr = util.read_processed_midi(n)
	for j in range(len(arr)):
		data.append(arr[j][1])
	data = np.array(data)
	data.view(np.long)
	return torch.from_numpy(data)

def gen(input):
	print("Generated text: ")
	model = util.load_model("models/pitch_model.pt")
	data = np.array(list(input))
	data = torch.from_numpy(data).long()
	generate.gen(model, data)

def prepare_data(read_path):
	# Uncomment this if you dont have the processed midi files
	#util.write_processed_midi(read_path)
	util.write_all_processed_midi_to_events()
	return

def main(read_path="", write_path="output.midi"):

	# RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
	# util.write_processed_midi(read_path)

	# Just testing
	#mid_data = util.read_processed_midi(0)
	#print(mid_data)
	# print("ROWS: ", mid_data.shape[0], "COLS: ", mid_data.shape[1])
	
	# Training the model
	params = [
		8, 	# n_heads
		4, 	# depth
		32, # seq_length
		128,# n_tokens 
		377, # emb_size 
		850,# n_batches 
		64, # batch_size 
		50, # test_every 
		0.000065, # lr 
		100,# warmup 
		-1 # seed
		]
	
	losses = generate.train(
		n_heads=params[0], 
		depth=params[1], 
		seq_length=params[2], 
		n_tokens=params[3], 
		emb_size=params[4], 
		n_batches=params[5], 
		batch_size=params[6], 
		test_every=params[7], 
		lr=params[8], 
		warmup=params[9], 
		seed=params[10],
		output_path="model.pt"
		)
	model = util.load_model("model.pt")
	util.save_on_train(model, losses, params[5], params, model_name=None)
	
	"""
	# Generate text

	gen([61, 65, 61, 73, 65])
	#, 55, 59, 62 print(util.load_all_predata_pitchonly(2))
	"""
	#print(get_pitches(900)[:10])
	# twinkle_midi = pretty_midi.PrettyMIDI(util.here("twinkle.midi")).instruments[0].notes

	# util.write_all_processed_midi_to_event_indices()
	# Get midi
	"""
	midi_array = util.read_processed_midi(0) # 8 is a good test case (16 errors)
	#midi_array = util.midi_to_array(util.here("testmidi/mii.midi"))
	#print("ORIGINAL LENGTH", len(midi_array))
	#print("ORIGINAL ARRAY", midi_array)
	# Convert to events
	midi_events = dataprocess.midi_array_to_event2(midi_array)

	# Convert to indices
	midi_event_indices = []
	for i in midi_events:
		midi_event_indices.append(dataprocess.event_to_index(i))
	# Convert back to events
	midi_events2 = []
	for i in midi_event_indices:
		midi_events2.append(dataprocess.index_to_event(i))
	# print("NEW EVENTS", midi_events2)
	# Convert back to midi
	midi_array2 = dataprocess.event_to_midi_array(midi_events2)
	#print("NEW ARRAY", midi_array2)
	#print("NEW LENGTH", len(midi_array2))
	# Save
	util.write_piano_midi(midi_array2, util.here("output.midi"))
	"""

if __name__ == '__main__':
	main(read_path="maestro-v2.0.0")
