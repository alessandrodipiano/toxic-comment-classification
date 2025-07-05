the project folder contains the following:


notebook:

ICTC: contains the project code 
NOTE: some cells of the code have very high memory requirements, and are not possible to execute in online editors;
	the output of those cells is saved as files that are re-loaded later,


pdf:
the report of the ICTC project 



sub-folders:

	matrixes: it contains the saved ppmi and tfidf matrices for computational efficiency, they can be loaded using np.load for inspection 

	indexes: contains the word-to-index dictionaries for both matrixes, to be loaded using pickle command 

	models: contains all the trained neural models: RNN with ppmi embeddings, 
				RNN with word2vec embeddings, 
				RNN with tf-idf embeddings,  
				RNN with own embeddings (subword),
				BERT and BERT tokenizer in the same folder.


text files:

requirement_txt: contains the required packages, together with the versions we used (needed for compatibility)

train_comments_txt: text of all comments from the training set merged into one .txt file, used for subword tokenization











