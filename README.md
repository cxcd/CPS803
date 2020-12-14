# CPS803 Final Project: Music Generation Using Transformers

Our final project for the Ryerson University course CPS803 Machine Learning, in which we apply our understanding of transformers to the problem of music generation. This project was largely based on the Peter Bloem transformer example [[1]](#1), and the dataset pre-processing was informed by the 2018 Music Transformer paper [[2]](#2). Due to time constraints the full innovations of the Music Transformer algorithm could not be implemented.


## Usage
This project was developed using Anaconda Navigator. With Navigator, the environment is automatically set up, however without it the environment can be created using the included evironment file:
```
conda env create -f environment.yml
```

To preprocess and prepare the dataset for use by the model, run the `prepare_data()` function in the main file once, with the path to the dataset passed as a parameter. This will create a folder containing all the processed data. This project was intended to be run with the entire dataset.
```
prepare_data("maestro-v2.0.0")
```

To train the model on the data, and save the results, run the following functions in the main function:
```
params = [
	8, 	  # n_heads
	4, 	  # depth
	32,	  # seq_length
	378,	  # n_tokens 
	64, 	  # emb_size 
	900,	  # n_batches 
	32, 	  # batch_size 
	50, 	  # test_every 
	0.000005, # lr 
	250,	  # warmup 
	-1 	  # seed
	]
	
losses = generate.train(
	n_heads=params[0], 
	depth=params[1], 
	seq_length=params[2], 
	n_tokens=params[3], 
	emb_size=params[4], 
	n_batches=params[5], 
	batch_size=params[6], 
	test_every=params[7], 
	lr=params[8], 
	warmup=params[9], 
	seed=params[10],
	output_path="model.pt"
	)
model = util.load_model("model.pt")
util.save_on_train(model, losses, params[5], params, model_name=None)
```
To generate a result from an existing model, run the following function in the main function:
```
# Input is an array of indices in the processed format described in the paper
gen(input)
```



## References

<a id="1">[1]</a> 
Huang, C. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., Dai, A. M., Hoffman, M. D., Dinculescu, M., & Eck, D. (2018). Music transformer: Generating music with long-term structure. CoRR. http://arxiv.org/abs/1809.04281

<a id="2">[2]</a> 
P. Bloem, former, (2019), GitHub repository, https://github.com/pbloem/former

