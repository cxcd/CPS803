import util
import generate
import torch
import numpy as np

def gen(input):
    print("Generated text: ")
    model = util.load_model("models/genmodel.pt")
    data = np.array(list(input))
    data = data.view(np.uint8)
    data = torch.from_numpy(data).long()
    generate.gen(model, data)

def main(read_path="", write_path="output.midi"):

    # RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
    # util.write_processed_midi(read_path)

    # Just testing
    # mid_data = util.read_processed_midi(0)
    # print(mid_data)
    # print("ROWS: ", mid_data.shape[0], "COLS: ", mid_data.shape[1])

    # Training the model
    generate.train(data="data/the.txt", output_path="models/genmodel.pt")

    # Generate text
    gen("the ")

if __name__ == '__main__':
    main(read_path="maestro-v2.0.0")
