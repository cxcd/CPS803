import os
import util

def main(read_path="", write_path="output.midi"):

    # RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
    # util.write_processed_midi(read_path)

    # Just testing
    mid_data = util.read_processed_midi(0)
    print(mid_data)
    print("ROWS: ", mid_data.shape[0], "COLS: ", mid_data.shape[1])

if __name__ == '__main__':
    main(read_path="maestro-v2.0.0")
