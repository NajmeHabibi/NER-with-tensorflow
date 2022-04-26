import pickle

import numpy as np
from keras_preprocessing.sequence import pad_sequences
import unittest

from database import NamedEntity
from lstm_ner.lstm_ner import LSTM


class BertNER:
    def __init__(self):
        from transformers import AutoTokenizer, TFAutoModelForTokenClassification
        from transformers import pipeline

        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = TFAutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self._ner = pipeline("ner", model=model, tokenizer=tokenizer)

        self.label_names = {'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}

    def _group_named_entities(self, nes):
        grouped_nes = []
        i = 0
        while i < len(nes):
            bne = nes[i]  # beginning word of a chunk
            j = i + 1
            while j < len(nes) and nes[j]['entity'].startswith('I'):
                ine = nes[j]  # next inside word of current chunk
                if ine['word'].startswith('##'):
                    bne['word'] = bne['word'] + ine['word'][2:]
                else:
                    bne['word'] = bne['word'] + ' ' + ine['word']
                bne['end'] = ine['end']
                j += 1
            i = j
            bne['entity'] = bne['entity'][2:]
            grouped_nes.append(bne)

        return grouped_nes

    def __call__(self, text, *args, **kwargs):
        grouped_nes = self._group_named_entities(nes=self._ner(text))

        for ne in grouped_nes:
            if ne['entity'] in self.label_names.keys():
                label_name = self.label_names[ne['entity']]
                yield NamedEntity(entity=ne['word'], start_char=ne['start'], end_char=ne['end'], label=label_name)


class SpacyNER:
    def __init__(self):
        import spacy

        self._ner = spacy.load("en_core_web_sm")  # "en_core_web_lg"

        self.label_names = {'PERSON': 'PER', 'ORG': 'ORG', 'GPE': 'LOC'}

    def __call__(self, text, *args, **kwargs):
        assert isinstance(text, str)

        for ent in self._ner(text).ents:
            if ent.label_ in self.label_names.keys():
                label_name = self.label_names[ent.label_]
                # yield {'entity': ent.text, 'start_char': ent.start_char, 'end_char': ent.end_char, 'label': label_name}
                yield NamedEntity(entity=ent.text, start_char=ent.start_char, end_char=ent.end_char, label=label_name)


class LstmNER:
    def __init__(self):
        self.lstm = LSTM()
        self.label_names = {'per': 'PER', 'org': 'ORG', 'geo': 'LOC'}

        self.model = pickle.load(open("lstm_ner/saved/lstm_model.pickle", 'rb'))
        self.tokenizer = pickle.load(open("lstm_ner/saved/tokenizer.pickle", 'rb'))

    def __call__(self, text, *args, **kwargs):
        grouped_nes = self._group_named_entities(nes=self.lstm.predict(text, self.model, self.tokenizer))
        # print(grouped_nes)
        for ne in grouped_nes:
            label_name = self.label_names[ne['entity']]
            yield NamedEntity(entity=ne['word'], start_char=ne['start'], end_char=ne['end'], label=label_name)

    def _group_named_entities(self, nes):
        grouped_nes = []
        i = 0
        while i < len(nes):
            bne = nes[i]  # beginning word of a chunk
            new_text = bne[0]
            start = bne[2]
            end = bne[3]
            j = i + 1
            while j < len(nes) and nes[j][1].startswith('I'):
                ine = nes[j]  # next inside word of current chunk
                new_text = new_text + ' ' + ine[0]
                end += ine[3]
                j += 1
            i = j
            grouped_nes.append({'word': new_text, 'entity': bne[1][2:], 'start': start, 'end': end})
        return grouped_nes
