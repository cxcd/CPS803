import util
import generate
import torch
import numpy as np
import util
import dataprocess
import os
import pretty_midi
from operator import itemgetter


def gen(input):
	print("Generated text: ")
	model = util.new_load_model()
	data = np.array(list(input))
	data = torch.from_numpy(data).long()
	index_arr = generate.gen(model, data, 700)
	event_arr = dataprocess.indices_to_events(index_arr)
	midi_arr = dataprocess.event_to_midi_array(event_arr)
	util.save_on_gen(input, midi_arr, model_num=None, gen_num=None)
	print(event_arr)

def prepare_data(read_path):
	# Uncomment this if you dont have the processed midi files
	#util.write_processed_midi(read_path)
	util.write_all_processed_midi_to_event_indices_augmented()
	return

def main(dataset_path="maestro-v2.0.0"):
	# RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
	#prepare_data(dataset_path)
	#return
	
	# Uncomment to use a specified file to generate
	"""
	index_arr = util.read_processed_event_index(1200)
	gen(index_arr[:20])
	return
	"""
	# Training the model
	
	# Uncomment to train model
	params = [
		8, 	# n_heads
		4, 	# depth
		32, # seq_length
		378,# n_tokens 
		64, # emb_size 
		900,# n_batches 
		32, # batch_size 
		50, # test_every 
		0.000005, # lr 
		250,# warmup 
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

if __name__ == '__main__':
	main()
