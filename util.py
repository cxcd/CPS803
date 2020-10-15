import os
import numpy as np
import pretty_midi

def read_all_midi(dir_path):
    data = []
    if(not os.path.exists('processed_midi_files')):
        fileNum = 0
        os.makedirs("processed_midi_files")
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".midi"):
                    temp = midi_to_array(os.path.join(root, file))
                    filename = 'midi_'+str(fileNum)+'.csv'
                    with open(os.path.join('\\processed_midi_files', filename), 'a+') as f:
                        for i in range(len(temp[0])):
                            np.savetxt(filename, temp[:,i], delimiter=',')
                    data.append(temp) #Could potentially output everything to one file
                    fileNum += 1
    else:
        for file in os.listdir('processed_midi_files'):
            data.append(np.loadtxt('midi.csv', delimiter=','))
    return data


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