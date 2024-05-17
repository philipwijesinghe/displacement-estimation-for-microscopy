# displacement-estimation-deep-learning
Displacement estimation in biological images using deep learning

## Example data, training and inference

Minimal example data

For this we use the Drosophila pulsing video with 86 frames at 1024x1024 pix

We include training data and code to regenerate it

We include a trained model, and an untrained model with a config file based on the training data

The inference folder shows how to organise real video data, and process it to displacements

```
./example/drosophila/Ref
					/Def
		 /training/drosophila/training/Ref/
		 							  /Def/
		 							  /Dispx/
		 							  /Dispy/
		 							  /IntMod/
		 							  /...
		 					 /testing/ "
		 					 /validation/ "
		 /models/example_model/saved_models/checkpoint.pth
		 					  /config.yml
		 					  /...
		 		/example_model_untrained/config.yml
		 /inference/drosophila/Ref
		 					  /Def
		 					  /Dispx-inference
		 					  /Dispy-inference
		 					  /...
```


