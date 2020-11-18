import util
import generate
import torch
import numpy as np

"""
# Loads only the pitches from preprocessed data
# If no parameters, load all the data
data = util.load_all_predata_pitchonly()

# Writes pitches to a midi file
util.write_piano_midi(util.pitches_to_midi(some_array))

"""

def gen(input):
    print("Generated text: ")
    model = util.load_model("models/the_model.pt")
    data = np.array(list(input))
    data = data.view(np.uint8)
    data = torch.from_numpy(data).long()
    generate.gen(model, data)

def main(read_path="", write_path="output.midi"):

    # RUN THIS FIRST TO GENERATE THE PROCESSED DATASET
    # util.write_processed_midi(read_path)




    # Just testing
    #mid_data = util.read_processed_midi(0)
    #print(mid_data)
    # print("ROWS: ", mid_data.shape[0], "COLS: ", mid_data.shape[1])
    """
    # Training the model
    generate.train(
        n_heads=8, 
        depth=4, 
        seq_length=32, 
        n_tokens=256, 
        emb_size=128, 
        n_batches=500, 
        batch_size=64, 
        test_every=50, 
        lr=0.0001, 
        warmup=100, 
        seed=-1,
        data="data/books.txt", 
        output_path="models/genmodel.pt"
        )
    """
    # Generate text
    # gen("the ")
    # print(util.load_all_predata_pitchonly(2))


if __name__ == '__main__':
    main(read_path="maestro-v2.0.0")
