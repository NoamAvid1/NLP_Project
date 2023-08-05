from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd


def get_df_scores():
    #todo: complete
    return


def get_bert_scores(model_seed_and_step, text, correct_word, false_word):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    unmasker = pipeline('fill-mask', model=model_seed_and_step, tokenizer=model_seed_and_step)
    filled = unmasker(text, targets=[correct_word, false_word])
    correct_word_score = filled[0]['score']
    false_word_score = filled[1]['score']
    return correct_word_score, false_word_score


def get_pythia_scores(model_name, model_revision, text, correct_word, false_word):
  """
  :param model_name: name including no. of params, e.g., "EleutherAI/pythia-70m-deduped"
  :param model_revision: num of training steps,e.g., "step3000"
  :return: scores of the correct next word and the false word, as floats
  """
  model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=model_revision,
    # cache_dir="./pythia-70m-deduped/step3000",
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=model_revision,
    # cache_dir="./pythia-70m-deduped/step3000",
  )

  inputs = tokenizer("Hello, I am", return_tensors="pt")
  probs = model(**inputs).logits[0][-1]  # todo: check that the last index should be -1
  correct_word_score = probs[tokenizer.encode(correct_word)][0].item()
  false_word_score = probs[tokenizer.encode(false_word)][0].item()
  return correct_word_score, false_word_score


if __name__ == '__main__':
    get_bert_scores('google/multiberts-seed_4-step_20k', "Hello I'm a [MASK] model.")