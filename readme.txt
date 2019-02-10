1. The training models are generated automatically after each training.For now
	- trained_model_1509637563 is the trained model for sentiment
	- trained_model_1508867627 is the trained model to detect deigital/ non-digital
	- This two models can be trained with a larger dataset anytime in the future
2. All labels file and parameters files are the json files where parameters are stored
	- This can be toned in everytime of new training
3. nps_stem_2.py is the main function where we impose the trained models and introduce the pickle files and make predictions
	- The path needs to be changed to local input files
	- Potentially it is possible to wirte an interface that allows this program to connect to Madellia directly.
	

Training Data looks at the following format
reviews			label
make things fucked up	negative
less technical issues	negative
No more etransfer fee	negative
If it ain't broken...	negative
 less outages please!	negative
 Not much it is great	negative
Poor servic		negative
