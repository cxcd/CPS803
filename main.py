import util
import generate
import torch
import numpy as np
import util

def gen(input):
	print("Generated text: ")
	model = util.load_model("models/pitch_model.pt")
	data = np.array(list(input))
	data = torch.from_numpy(data).long()
	output = generate.gen(model, data)
	util.write_piano_midi(util.pitchesvelocity_to_midi(output), "output2.midi")

def prepare_data(read_path):
	util.write_processed_midi(read_path) # COMMENT THIS OUT IF YOU ALREADY HAVE THE OLD PROCESSED DATA
	util.write_all_processed_midi_to_event_indices()
	return

def main(read_path="", write_path="output.midi"):
	
	# RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
	"""
	prepare_data(read_path)
	return
	"""

	# Training the model
	params = [
		8, 	# n_heads
		4, 	# depth
		32, # seq_length
		378,# n_tokens 
		128, # emb_size 
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

if __name__ == '__main__':
	main(read_path="maestro-v2.0.0")
