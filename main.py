import os
import numpy as np
import pretty_midi
import util

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

def main(read_path="", write_path="output.midi", notes=twinkle_notes):

    # RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
    util.write_processed_midi(read_path)

    # util.write_piano_midi(notes, write_path)
    # print(util.read_processed_midi(0))

if __name__ == '__main__':
    main(read_path="maestro-v2.0.0")
