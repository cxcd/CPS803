import os
import numpy as np
import pretty_midi
import torch
import dataprocess
import matplotlib.pyplot as plt
import dataprocess

# Directory to load processed midi files
processed_dir = 'processed_midi_files\\'
# Directory to load processed event files
processed_events_dir = 'processed_event_indices_files/'
# Number of files in the data set
max_files = 1281

"""
0 - 999 train
1000 - 199 - valid
1200 - 1280 - train
"""

# Twinkle twinkle example midi
twinkle_notes = np.array([
		# C major
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C4'), start=0, end=1),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('E4'), start=0, end=1),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G4'), start=0, end=1),
		# Twinkle, twinkle
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C5'), start=0, end=0.25),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C5'), start=0.25, end=0.5),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G5'), start=0.5, end=0.75),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G5'), start=0.75, end=1),
		# F major
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('F4'), start=1, end=2),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G4'), start=1, end=2),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C5'), start=1, end=2),
		# Little
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('A5'), start=1, end=1.25),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('A5'), start=1.25, end=1.5),
		# C major
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C4'), start=1.5, end=2),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('E4'), start=1.5, end=2),
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G4'), start=1.5, end=2),
		# Star
		pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G5'), start=1.5, end=2),
		])

def device(tensor=None):
	"""
	Get the available computaiton device
	"""
	return 'cuda' if torch.cuda.is_available() else 'cpu'

def use_cuda():
	return torch.cuda.is_available()

def here(file_name):
	"""
	Get the given file name relative to the working directory
	"""
	return os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

def pitches_to_midi(pitches, duration=0.25, velocity=100):
	"""
	Turn array of pitches into full midi data
	"""
	notes = []
	time_acc = 0
	for i in range(len(pitches)):
		notes.append(pretty_midi.Note(velocity, int(pitches[i]), time_acc, time_acc + duration))
		time_acc += duration
	return np.array(notes)

def pitchesvelocity_to_midi(data, duration=0.25, velocity=100):
	"""
	Turn array of pitches into full midi data
	"""
	notes = []
	time_acc = 0
	for i in range(len(data)):
		notes.append(pretty_midi.Note(int(data[i][0]), int(data[i][1]), time_acc, time_acc + duration))
		time_acc += duration
	return np.array(notes)

def array_to_midi(input):
	"""
	Convert array to the pretty_midi format
	"""
	notes = []
	for note in input:
		notes.append(pretty_midi.Note(int(note[0]), int(note[1]), note[2], note[3]))
	return np.array(notes)

def midi_to_array(midi_path):
	"""
	Read MIDI file into a 4D array where each element is [velocity, pitch, start, end]
	"""
	# Get MIDI data
	data = pretty_midi.PrettyMIDI(midi_path).instruments[0].notes
	# Init 4D array
	array = np.zeros((len(data),4))
	# Add MIDI data to array
	for i in range(len(data)):
		array[i] = ([data[i].velocity, data[i].pitch, data[i].start, data[i].end])
	# Return array
	return array

def write_piano_midi(notes, write_path):
	"""
	Output an array of notes to the desired path as piano MIDI
	"""
	# Create the output structure
	output = pretty_midi.PrettyMIDI()
	# Create the instrument program and instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	# Set the piano notes to the list of notes
	piano.notes = notes
	# Give the output our instrument
	output.instruments.append(piano)
	# Write the output
	output.write(write_path)

def write_processed_midi(dataset_path):
	"""
	Process all the midi into text data
	"""
	data = []
	file_num = 0
	curr_dir = os.path.dirname(__file__)
	# If the directory holding the processed files doesn't exist
	if(not os.path.exists(processed_dir)):
		os.mkdir(processed_dir)
	# Perform a walk in the dataset directory
	for root, dirs, files in os.walk(dataset_path):
		# For each file in the directory and subdirectories
		for file in files:
			# For all midi files
			if file.endswith(".midi"):
				print("Util: Saving midi_", file_num, "...")
				# Grab the midi and generate a file name
				temp = midi_to_array(os.path.join(root, file))
				file_name = os.path.join(curr_dir, processed_dir + 'midi_'+str(file_num))
				# Save the file
				np.save(file_name, temp)
				# Increment the file number
				file_num += 1
	print("Util: Finished saving MIDI to array.", file_num + 1, "total files.")
	return data

def read_processed_midi(file_num):
	"""
	Read a processed midi file given its file number
	"""
	if (file_num <= max_files and file_num >= 0):
		curr_dir = os.path.dirname(__file__)
		path = os.path.join(curr_dir, processed_dir + 'midi_'+str(file_num) + ".npy")
		return np.load(path)

def load_all_predata(n=None):
	data = []
	# Set the range
	if (n is None) or (n > max_files):
		n = max_files
	# Get the data
	for i in range(n):
		data.append(read_processed_midi(i))
	data = np.array(data)
	data.view(np.float)
	return torch.from_numpy(data)

def load_all_predata_event_indices(n=None):
	data = []
	# Set the range
	if (n is None) or (n > max_files):
		n = max_files
	# Get the data
	for i in range(n+1):
		data = np.append(data, read_processed_event_index(i))
	data = np.array(data)
	data.view(np.float)
	return data

def load_all_predata_pitchonly(n=None):
	data = []
	# Set the range
	if (n is None) or (n > max_files):
		n = max_files
	# Get the data
	for i in range(n):
		arr = read_processed_midi(i)
		for j in range(len(arr)):
			data.append(arr[j][1])
	data = np.array(data)
	data.view(np.float)
	return torch.from_numpy(data)

def load_all_predata_pitchvelocityonly(n=None):
	data = []
	# Set the range
	if (n is None) or (n > max_files):
		n = max_files
	# Get the data
	for i in range(n):
		arr = read_processed_midi(i)
		for j in range(len(arr)):
			data.append([arr[j][0], arr[j][1]])
	data = np.array(data)
	data.view(np.float)
	return torch.from_numpy(data)

def save_model(model, path):
	"""
	Save the whole PyTorch model
	"""
	torch.save(model, here(path))

def load_model(path):
	"""
	Load the whole PyTorch model
	"""
	model = torch.load(here(path))
	model.eval()
	return model

# A variable to determine the directory to store the models in
models_path = 'models/'

#Creates a directory at the specified path
def create_dir(dir_path):
	if(not os.path.exists(dir_path)):
		os.mkdir(dir_path)

#Gets the latest model number in the models directory
def get_latest_model_num():
	if(not os.path.exists(models_path)):
		create_dir(models_path)
	model_num = 0
	for root, dirs, files, in os.walk(models_path):
		for dir in dirs:
			if dir.startswith('model_'):
				temp_num = int(dir[len('model_'):])
				if temp_num > model_num:
					model_num = temp_num
	return model_num

#Gets the latest gen_num, if model_num is not given the latest model_num will be given
def get_latest_gen_num(model_num=None):
	if model_num==None:
		model_num = get_latest_model_num()
	gen_num = 0
	for root, dirs, files, in os.walk(models_path+'model_'+str(model_num)+'/gens/'):
		for file in files:
			if file.endswith('.midi'):
				temp_num = int(file[len('gen_'):-len('.midi')])
				if temp_num > gen_num:
					gen_num = temp_num
	return gen_num

# Creates a loss vs epoch plot, Creates a directory for the current model,
# then saves the model, plot, loss, and params for the model
# params - An array containing all the params in order
# losses - An array containing all the losses
# num_epoch - Number of epochs
# model_name must follow format trained_model_x.pt, where x is an integer 
def save_on_train(model, losses, num_epochs, params, model_name=None):
	create_dir(models_path)
	model_num = 0
	if model_name==None:
		model_num = get_latest_model_num()
		if(os.path.exists(models_path+'model_'+str(model_num))):
			model_num += 1
		model_name = 'trained_model_'+str(model_num)+'.pt'
		dir_path = models_path+'model_'+str(model_num)+'/'
	else:
		if model_name[-3:] != ".pt" or model_name[:len("trained_model_")] != "trained_model_" or not model_name.replace("trained_model_","")[:-3].isdigit():
			print("Error: model_name is incorrect and does not follow the format \"trained_model_x.pt\" where x is an integer")
			return 
		model_num = model_name.replace("trained_model_","")[:-len('.pt')]
		dir_path = models_path+'model_'+model_num+'/'
	create_dir(dir_path)
	create_dir(dir_path+'gens/')
	x_axis = np.arange(num_epochs)
	fig, (loss_plot) = plt.subplots(1,1)
	loss_plot.plot(x_axis, losses)
	loss_plot.set_xlabel('epochs')
	loss_plot.set_ylabel('loss')
	loss_plot.legend()
	fig.savefig(dir_path+'loss.png')
	np.savetxt(dir_path+'losses_'+str(model_num)+'.txt', losses)
	np.savetxt(dir_path+'params_'+str(model_num)+'.txt', params)
	torch.save(model, dir_path+model_name)

# Saves the input and midi_file to a folder
# Will save to the latest model, and will create new gen file if no gen_num is specified
# Specified gen_num will overwrite a file 
# Specified model_num will be saved into that model
def save_on_gen(input, midi_file, model_num=None, gen_num=None):
	if model_num==None:
		model_num = get_latest_model_num()
	if gen_num==None:
		gen_num = get_latest_gen_num(model_num)
		if(os.path.exists(models_path+'model_'+str(model_num)+'/gens/gen_'+str(gen_num)+'_input.npy')):
			gen_num += 1
	dir_path = models_path+'model_'+str(model_num)+'/gens/'
	np.savetxt(dir_path+'gen_'+str(gen_num)+'_input.txt', input)
	write_piano_midi(midi_file, dir_path+'gen_'+str(gen_num)+'.midi')


# Returns empty list if it does not exist
# Grabs the params of the specified model number
def load_param(model_num):
	param_path = models_path+'model_'+str(model_num)+'/params_'+str(model_num)+'.txt'
	if(os.path.exists(param_path)):
		return np.loadtxt(param_path)
	return []

# Returns empty list if it does not exist
# Grabs the model of the specified model number
def new_load_model(model_num):
	model_path = models_path+'model_'+str(model_num)+'/trained_model_'+str(model_num)+'.pt'
	if(os.path.exists(model_path)):
		return torch.load(model_path)
	return []

def write_all_processed_midi_to_event_indices():
	"""
	Takes all processed midi, processes it into events, then indices and saves them into the processed_indices_dir
	"""
	create_dir(processed_events_dir) 
	for i in range(max_files+1):
		midi_arr = read_processed_midi(i)
		event_arr = dataprocess.midi_array_to_event2(midi_arr)
		index_arr = dataprocess.events_to_indices(event_arr)
		np.save(processed_events_dir+'event_index_arr_'+str(i)+'.npy', index_arr)
		print("Saving file", i, "...")
	print("Complete!")

def read_processed_event_index(file_num):
	"""
	Read a processed event file given its file number
	"""
	return np.load(processed_events_dir+'event_index_arr_'+str(file_num)+'.npy')

