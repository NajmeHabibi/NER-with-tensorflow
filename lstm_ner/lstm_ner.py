import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pathlib


class LSTM:
    def __init__(self):
        self.data_loader = DataLoader()

    def train(self):
        pathlib.Path("saved/").mkdir(parents=True, exist_ok=True)

        train_dataset, val_dataset, test_dataset = self.data_loader.get_data()
        train_set = tf.data.Dataset.from_tensor_slices(train_dataset)
        val_set = tf.data.Dataset.from_tensor_slices(val_dataset)
        test_set = tf.data.Dataset.from_tensor_slices(test_dataset)
        BATCH_SIZE = 132
        SHUFFLE_BUFFER_SIZE = 132
        train_dataset = train_set.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        val_dataset = val_set.batch(BATCH_SIZE)
        test_dataset = test_set.batch(BATCH_SIZE)
        embedding_dim = 300
        maxlen = self.data_loader.max_len
        max_words = 60000
        num_tags = len(self.data_loader.tag_list)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax'))
        ])
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=1)

        pickle.dump(model, open('saved/lstm_model.pickle', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        model.evaluate(test_dataset)
        return history

    # def get_pretrained_model(self):
    #     return pickle.load(open("lstm_ner/saved/lstm_model.pickle", 'rb'))
    #
    # def get_tokenizer(self):
    #     return pickle.load(open("lstm_ner/saved/tokenizer.pickle", 'rb'))

    def predict(self, text, model, tokenizer):
        original_news = text.split(".")
        tokenized_texts = tokenizer.texts_to_sequences(original_news)
        processed_X = pad_sequences(tokenized_texts, maxlen=110, padding='post')
        nes = []
        current_start = 0
        for i, x in enumerate(processed_X):
            prediction = model.predict(np.asarray([x]))
            prediction = np.argmax(prediction[0], axis=1)
            original_list = original_news[i].split()
            prediction = list(prediction)[: len(original_list)]
            for j, predict in enumerate(prediction):
                word = original_list[j]
                tag = self.data_loader.reversed_tag_list[predict]
                if tag != 'O':
                    nes.append((word, tag, current_start, current_start + len(word)))
                current_start += len(word) + 1
        return nes


class DataLoader:
    data_path = "data/ner.csv"
    tag_list = {'O': 0, 'B-geo': 1, 'B-per': 2, 'I-geo': 3, 'B-org': 4, 'I-org': 5, 'I-per': 6}
    reversed_tag_list = {v: k for k, v in tag_list.items()}
    max_len = 110

    def load(self):
        data_frame = pd.read_csv(self.data_path, encoding='unicode_escape')
        X = list(data_frame['Sentence'])
        Y = list(data_frame['Tag'])
        return X, Y

    def fit_tokenizer(self):
        ner_frame = pd.read_csv("data/ner.csv", encoding='unicode_escape')
        ner_frame.fillna(method='ffill', inplace=True)
        cnbc_frame = pd.read_csv("data/cnbc_news_datase.csv",
                                 encoding='unicode_escape')
        cnbc_frame.fillna(method='ffill', inplace=True)
        cbc_frame = pd.read_csv("data/cbc.csv", encoding='unicode_escape')
        cbc_frame.fillna(method='ffill', inplace=True)

        max_words = 60000
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(list(ner_frame.Sentence))
        tokenizer.fit_on_texts(list(cnbc_frame.short_description))
        tokenizer.fit_on_texts(list(cbc_frame.news))
        pickle.dump(tokenizer, open('saved/tokenizer.pickle', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

        return tokenizer

    def Y_preprocessing(self, Y):
        tokenized_Y = []
        for label in Y:
            new_Y = []
            for i, tag in enumerate(eval(label)):
                if tag in self.tag_list:
                    new_Y.append(self.tag_list[tag])
                else:
                    new_Y.append(0)
            tokenized_Y.append(new_Y)
        processed_Y = pad_sequences(tokenized_Y, maxlen=self.max_len, padding='post')
        return np.asarray(processed_Y)

    def X_preprocessing(self, X):
        tokenizer = self.fit_tokenizer()
        tokenized_texts = tokenizer.texts_to_sequences(X)
        processed_X = pad_sequences(tokenized_texts, maxlen=self.max_len, padding='post')
        return np.asarray(processed_X)

    def split_data(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_data(self):
        X, Y = self.load()
        processed_X = self.X_preprocessing(X)
        processed_Y = self.Y_preprocessing(Y)
        return self.split_data(processed_X, processed_Y)


if __name__ == '__main__':
    LSTM().train()