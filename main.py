import os
from numba import jit
import numpy as np
import pretty_midi

x = np.arange(100).reshape(10, 10)

test_notes = np.array([
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('C5'), start=0, end=1),
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('Eb5'), start=1, end=2),
        pretty_midi.Note(velocity=100, pitch=pretty_midi.note_name_to_number('G5'), start=2, end=3)
        ])

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting


def read_midi(midi_path):
    midi_data = pretty_midi.PrettyMIDI('example.midi')
    for i in midi_data.instruments[0].notes:
        print(i)


# Output an array of notes to the desired path
def make_piano_midi(notes, output_path):
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
    output.write(output_path)

if __name__ == '__main__':
    #read_midi(midi_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.midi'))
    make_piano_midi(test_notes, 'output.midi')
