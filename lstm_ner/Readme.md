# Custom LSTM Model Trained for Named Entity Recognition:

* Our custom model `lstm_ner.LSTM` for named entity recognition is designed using `tensorflow.keras`, 
* Trained and evaluated on CONLL (Sang and Meulder, 2003) dataset, 
* Model and Tokenizer of which are saved under `saved` folder after training for further evaluation on crawled news dataset (will be used in `..ners.LstmNER`)
* Model achieved accuracy of %98.58 on train and %97.42 on test datasets

### Run:
In a console with an activated python environment, under current directory:
* Enter `pip install -r ../requirements.txt` to install dependencies
* Enter `python main.py` to train the model

### Unit Test:
* In order to run a unittest, enter `python lstm_ner.py` in the console