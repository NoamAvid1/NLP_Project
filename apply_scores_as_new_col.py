import pandas
from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd
import json


def get_correct_and_false_tokens(filled, row):
    if row['num_of_masks'] == 1:
        correct_tokens = [i for i in filled if i['token_str'] == row['correct_answer']]
        false_tokens = [i for i in filled if i['token_str'] == row['false_answer']]
    else:
        correct_tokens = [i for i in filled[1] if i['token_str'] == row['correct_answer']]
        false_tokens = [i for i in filled[0] if i['token_str'] == row['false_answer']]
    return correct_tokens, false_tokens

def get_df_bert_scores(model_name, model_revision, row):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    unmasker = pipeline('fill-mask', model=f"{model_name}-{model_revision}", tokenizer=f"{model_name}-{model_revision}")
    filled = unmasker(row['bert_question'], targets=[row['correct_answer'], row['false_answer']])
    # check that the correct and false words are in the dicitionary
    correct_tokens, false_tokens = get_correct_and_false_tokens(filled, row)
    if len(correct_tokens) == 0 or len(false_tokens) == 0:
        return pd.NA, pd.NA
    correct_word_score = correct_tokens[0]['score']
    false_word_score = false_tokens[0]['score']
    return correct_word_score, false_word_score


def get_df_pythia_scores(model_name, model_revision, row):
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
    inputs = tokenizer(row['pythia_question'], return_tensors="pt")
    scores = model(**inputs).logits[0][-1]
    probs = scores.softmax(dim=0)
    # check if the answers are in the dictionary:
    if len(tokenizer.encode(row['correct_answer'])) > 1 or len(tokenizer.encode(row['false_answer'])) > 1:
        return pd.NA, pd.NA
    false_word_score = probs[tokenizer.encode(row['false_answer'])[0]].item()
    if row['num_of_masks'] == 1:
        correct_word_score = probs[tokenizer.encode(row['correct_answer'])[0]].item()
    else:
        new_inputs = tokenizer(row['pythia_question'] + "the ", return_tensors="pt")
        new_scores = model(**new_inputs).logits[0][-3]  # todo: fix
        new_probs = new_scores.softmax(dim=0)
        correct_word_score = new_probs[tokenizer.encode(row['correct_answer'])[0]].item()
    return correct_word_score, false_word_score


def run_model_and_save(model_type, models_config, df):
    """
    :param model_type: Bert or Pythia
    :return:
    """
    for model in models_config[model_type]['models']:
        for checkpoint in models_config[model_type]['checkpoints']:
            model_df = df.copy()
            if model_type == "bert":
                model_df[['correct_answer_score', 'false_answer_score']] = model_df.apply(
                    lambda row: get_df_bert_scores(model, checkpoint, row), axis=1, result_type='expand')
            else:
                model_df[['correct_answer_score', 'false_answer_score']] = model_df.apply(
                    lambda row: get_df_pythia_scores(model, checkpoint, row), axis=1, result_type='expand')
            file_name = f'output_{model.replace("/", "_")}_{checkpoint}.csv'
            model_df.to_csv(file_name, index=False)
            print(f"saved result to {file_name}")


if __name__ == '__main__':
    # df = pd.read_csv('experiment_sentences_preprocessed_15.8.csv')
    df = pd.read_excel("demo.xlsx")
    # df = df[df["num_of_masks"] == 1]
    # df[['correct_answer_score_bert','false_answer_score_bert']] = df.apply(lambda row:get_df_bert_scores('google/multiberts-seed_4-step_20k', row), axis=1, result_type='expand')
    # df[['correct_answer_score_pythia','false_answer_score_pythia']] = df.apply(lambda row:get_df_pythia_scores("EleutherAI/pythia-70m-deduped","step20000", row), axis=1, result_type='expand')
    # df.to_excel('output_15_8.xlsx', index=False)
    models_config = {}
    with open("models_configs/all_run.json") as f:
            models_config = json.load(f)
    for model_type in models_config.keys():
        run_model_and_save(model_type, models_config, df)

#     "models": ["EleutherAI/pythia-70m","EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-12b"],