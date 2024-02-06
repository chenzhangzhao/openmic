##
Place the dataset compressed file in the raw folder of openmicdata and extract the compressed file into an openmic-2018 file within that folder
###
Then configure the parameters in config.py, and select true or false for the fragment to indicate whether data augmentation is required (which will increase processing time). Then, run it in the main directory Python - m maincode. preprocess to preprocess the audio,Preprocessing mainly involves resampling, data augmentation(optionall), extracting embeddings using vggish, and storing them as tfrecord files
##
Then run Python - m maincode. train training in the main directory, and the training results can be seen in the output.
 Before training, you can configure the training parameters in config.py