from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd



def encode_in_bert(list_of_words):
    # tokenizer = AutoTokenizer.from_pretrained('google/multiberts-seed_4-step_20k') 
    # for word in list_of_words:
    #     tokenizer.encode(word)
    model_seed_and_step = 'google/multiberts-seed_4-step_20k'
    unmasker = pipeline('fill-mask', model=model_seed_and_step, tokenizer=model_seed_and_step)
    text =' Hi I am [MASK]'
    filled = unmasker(text, targets =list_of_words) ## The output log is our indicate here


def read_correct_false_words(df):
    words_col_correct = df['correct_answer'].tolist()
    words_col_false = df['false_answer'].tolist()
    return words_col_correct + words_col_false


if __name__ == '__main__':
    df = pd.read_csv('preprocessed_data/experiment_sentences_preprocessed - All merged.csv')
    words_list = read_correct_false_words(df)
    print(encode_in_bert(words_list))
