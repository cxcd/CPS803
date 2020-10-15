import os
import numpy as np
import pretty_midi

# Directory to load processed midi files
processed_dir = 'processed_midi_files\\'
# Number of files in the data set
max_files = 1281

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
    print("Util: Finished saving midi as array files.")
    return data

# Read a processed midi file given its file number
def read_processed_midi(file_num):
    if (file_num <= max_files and file_num >= 0):
        curr_dir = os.path.dirname(__file__)
        path = os.path.join(curr_dir, processed_dir + 'midi_'+str(file_num) + ".npy")
        return np.load(path)