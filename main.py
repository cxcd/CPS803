import os
from numba import jit
import numpy as np
import pretty_midi

test_notes = np.array([
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C5'), start=0, end=1),
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('Eb5'), start=1, end=2),
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G5'), start=2, end=3)
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

def main(read_path="", write_path="", notes=None):
    # print(midi_to_array(read_path))
    write_piano_midi(notes, write_path)

if __name__ == '__main__':
    main(read_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.midi'), notes=test_notes, write_path='output.midi')