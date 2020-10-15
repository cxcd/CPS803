import os
import numpy as np
import pretty_midi

processed_dir = 'processed_midi_files/'

# Read MIDI file into a 4D array where each element is [start, end, pitch, velocity]
def midi_to_array(midi_path):
    # Get MIDI data
    data = pretty_midi.PrettyMIDI(midi_path).instruments[0].notes
    # Init 4D array
    array = np.zeros((len(data),4))
    # Add MIDI data to array
    for i in range(len(data)):
        array[i] = ([data[i].start, data[i].end, data[i].pitch, data[i].velocity])
    # Return array
    return array

# Output an array of notes to the desired path as MIDI
def write_piano_midi(notes, write_path):
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

# Process all the midi into text data
def write_processed_midi(dataset_path):
    data = []
    fileNum = 0
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
                # Grab the midi and generate a file name
                temp = midi_to_array(os.path.join(root, file))
                filename = os.path.join(curr_dir, processed_dir + 'midi_'+str(fileNum))
                # Save the file
                np.save(filename, temp)
                # Increment the file number
                fileNum += 1
    return data

def read_processed_midi(path):
    return 0