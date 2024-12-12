from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import pandas as pd
import torch
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import login, notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

from natasha import MorphVocab, Doc
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

import numpy as np

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
login(token='USE YOUR HUGGINGFACE TOKEN HERE')



pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    
)

class NounNumberChanger(object):
    def __init__(self):
        self.llm = pipeline

    def change_word(self, word):

        messages = [
                    {"role": "system", 
                     "content": 
                    """
                    Измени число этого существительного. Если оно было в единственном числе, напиши его же во множественном. 
                    Если оно во множественном числе, напиши его же в единственном.
                    В качестве ответа напиши ТОЛЬКО результат - данное существительное в нужной форме. Твой ответ должен состоять из одного слова.
                
                    # Пример:
                    Input: яблоко
                    Ответ: яблоки
                
                    Input: корабли
                    Ответ: корабль
                    """},
                    {"role": "user", "content":
                        f"""
                        Существительное: {word}
                        Ответ:
                        """}
                ]
        outputs = self.llm(
                            messages,
                            max_new_tokens=100,
                            pad_token_id = pipeline.tokenizer.eos_token_id
                        )
        return outputs[0]["generated_text"][-1]['content']
    def __call__(self, word):
        return self.change_word(word)


class NounCaseChanger(object):
    def __init__(self):
        self.llm = pipeline

    def change_word(self, word):

        messages = [
                    {"role": "system", 
                     "content": 
                    """
                    Измени падеж этого существительного. Произвольно выбери другой падеж.
                    В качестве ответа напиши ТОЛЬКО результат - данное существительное в изменённой форме. Твой ответ должен состоять из одного слова.
                
                    # Пример:
                    Input: яблоко
                    Ответ: яблоку
                
                    Input: кораблей
                    Ответ: корабли

                    Input: вафля
                    Ответ: вафлей
                    """},
                    {"role": "user", "content":
                        f"""
                        Существительное: {word}
                        Ответ:
                        """}
                ]
        outputs = self.llm(
                            messages,
                            max_new_tokens=100,
                            pad_token_id = pipeline.tokenizer.eos_token_id
                        )
        return outputs[0]["generated_text"][-1]['content']
    def __call__(self, word):
        return self.change_word(word)


class VerbTimeChanger(object):
    def __init__(self):
        self.llm = pipeline

    def change_word(self, word):

        messages = [
                    {"role": "system", 
                     "content": 
                    """
                    Измени время этого глагола. Произвольно выбери другое время.
                    В качестве ответа напиши ТОЛЬКО результат - данное существительное в изменённой форме. Твой ответ должен состоять из одного слова. Только одно слово без лишних символов.
                
                    # Пример:
                    Input: далаю
                    Ответ: буду делать
                
                    Input: бежала
                    Ответ: побежит

                    Input: красит
                    Ответ: красил
                    """},
                    {"role": "user", "content":
                        f"""
                        Глагол: {word}
                        Ответ:
                        """}
                ]
        outputs = self.llm(
                            messages,
                            max_new_tokens=100,
                            pad_token_id = pipeline.tokenizer.eos_token_id
                        )
        return outputs[0]["generated_text"][-1]['content']
    def __call__(self, word):
        return self.change_word(word)


class VerbNumberChanger(object):
    def __init__(self):
        self.llm = pipeline

    def change_word(self, word):

        messages = [
                    {"role": "system", 
                     "content": 
                    """
                    Измени число этого глагола. Если он был в единственном числе, измени его на множественное и наоборот.
                    В качестве ответа напиши ТОЛЬКО результат - данное существительное в изменённой форме. Твой ответ должен состоять из одного слова. Только одно слово без лишних символов.
                
                    # Пример:
                    Input: далаю
                    Ответ: делают
                
                    Input: загорала
                    Ответ: загорали

                    Input: победит
                    Ответ: победят
                    """},
                    {"role": "user", "content":
                        f"""
                        Глагол: {word}
                        Ответ:
                        """}
                ]
        outputs = self.llm(
                            messages,
                            max_new_tokens=100,
                            pad_token_id = pipeline.tokenizer.eos_token_id
                        )
        return outputs[0]["generated_text"][-1]['content']
    def __call__(self, word):
        return self.change_word(word)


class SpeechPartsDetector:
    def __init__(self):
        pass

    def noun_detect(self, text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        tags = []
        for i, token in enumerate(doc.tokens):    
            if token.pos == 'NOUN':
                tags.append(i)
        return tags, [d.text for d in doc.tokens]

    def verb_detect(self, text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        tags = []
        for i, token in enumerate(doc.tokens):    
            if token.pos == 'VERB':
                tags.append(i)
        return tags, [d.text for d in doc.tokens]

class GrammarSpoilerRU(object):
    def __init__(self):
        self.vnchanger = VerbNumberChanger()
        self.vtchanger = VerbTimeChanger()
        self.nnchanger = NounNumberChanger()
        self.ncchanger = NounCaseChanger()

        self.noun_changers = [self.nnchanger, self.ncchanger]
        self.verb_changers = [self.vnchanger, self.vtchanger]

        self.detector = SpeechPartsDetector()

    def spoil_verbs(self, text, spoil_proba=0.5):
        verbs, tokens = self.detector.verb_detect(text)
        for v_id in verbs:
            spoil_flag = np.random.choice([0, 1], p=[1-spoil_proba, spoil_proba])
            if spoil_flag:
                changer = np.random.choice(self.verb_changers)
                tokens[v_id] = changer(tokens[v_id])
        return ' '.join(tokens)

    def spoil_nouns(self, text, spoil_proba=0.5):
        nouns, tokens = self.detector.noun_detect(text)
        for n_id in nouns:
            spoil_flag = np.random.choice([0, 1], p=[1-spoil_proba, spoil_proba])
            if spoil_flag:
                changer = np.random.choice(self.noun_changers)
                tokens[n_id] = changer(tokens[n_id])
        return ' '.join(tokens)
