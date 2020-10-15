import os
import util

def main(read_path="", write_path="output.midi"):

    # RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
    util.write_processed_midi(read_path)

    #util.write_piano_midi(util.twinkle_notes, write_path)
    #print(util.read_processed_midi(0))

if __name__ == '__main__':
    main(read_path="maestro-v2.0.0")
