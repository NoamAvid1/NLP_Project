from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd

def get_bert_scores(model_seed_and_step, series):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    unmasker = pipeline('fill-mask', model=model_seed_and_step, tokenizer=model_seed_and_step)
    filled = unmasker(series['sentence_to_bert'], targets=[series['correct_answer'], series['false_answer']])
    correct_word_score = filled[0]['score']
    false_word_score = filled[1]['score']
    return correct_word_score, false_word_score

if __name__ == '__main__':
    series = {'sentence_to_bert': "Hello I'm a [MASK] model.", 'correct_answer': "model", 'false_answer': "orange"}
    df = pd.read_excel('demo.xlsx')
    df[['correct_answer_score','false_answer_score']] = df.apply(lambda row:get_bert_scores('google/multiberts-seed_4-step_20k',row), axis=1,result_type='expand' )
    df.to_excel('demo_after_applying.xlsx', index=False)
