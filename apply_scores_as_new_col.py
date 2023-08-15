import pandas
from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd


def get_correct_and_false_tokens(filled, row):
    if row['num_of_masks'] == 1:
        correct_tokens = [i for i in filled if i.token_str == row['correct_answer']]
        false_tokens = [i for i in filled if i.token_str == row['false_answer']]
    else:
        correct_tokens = [i for i in filled[1] if i.token_str == row['correct_answer']]
        false_tokens = [i for i in filled[0] if i.token_str == row['false_answer']]
    return correct_tokens, false_tokens

def get_df_bert_scores(model_seed_and_step, row):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    unmasker = pipeline('fill-mask', model=model_seed_and_step, tokenizer=model_seed_and_step)
    filled = unmasker(row['bert_question'], targets=[row['correct_answer'], row['false_answer']])
    # check that the correct and false words are in the dicitionary
    correct_tokens, false_tokens = get_correct_and_false_tokens(filled, row)
    if len(correct_tokens) == 0 or len(false_tokens) == 0:
        return pd.NA, pd.NA
    correct_word_score = correct_tokens[0]['score']
    false_word_score = false_tokens[0]['score']
    return correct_word_score, false_word_score


def get_df_pythia_scores(model_name, model_revision, series):
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        # cache_dir=f"./{model_name}/{model_revision}",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        # cache_dir=f"./{model_name}/{model_revision}",
    )
    inputs = tokenizer(series['pythia_question'], return_tensors="pt")
    scores = model(**inputs).logits[0][-1]
    probs = scores.softmax(dim=0)
    if len(tokenizer.encode(series['correct_answer'])) > 1 or len(tokenizer.encode(series['false_answer'])) > 1:
        return pd.NA, pd.NA
    correct_word_score = probs[tokenizer.encode(series['correct_answer'])[0]].item()
    false_word_score = probs[tokenizer.encode(series['false_answer'])[0]].item()
    return correct_word_score, false_word_score


if __name__ == '__main__':
    df = pd.read_csv('experiment_sentences_preprocessed_15.8.csv')
    df = df[df["num_of_masks"] == 1]
    df[['correct_answer_score_bert','false_answer_score_bert']] = df.apply(lambda row:get_df_bert_scores('google/multiberts-seed_4-step_20k', row), axis=1, result_type='expand')
    df[['correct_answer_score_pythia','false_answer_score_pythia']] = df.apply(lambda row:get_df_pythia_scores("EleutherAI/pythia-70m-deduped","step20000", row), axis=1, result_type='expand')
    df.to_excel('output_15_8.xlsx', index=False)
