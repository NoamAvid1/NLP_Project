from transformers import BertTokenizer, BertModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd



def get_bert_scores(model_seed_and_step, text, correct_word, false_word, num_of_masks):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    unmasker = pipeline('fill-mask', model=model_seed_and_step, tokenizer=model_seed_and_step)
    filled = unmasker(text, targets=[correct_word, false_word])
    print(filled)
    if num_of_masks == 1:
        correct_word_score = [i for i in filled if i.token_str == correct_word][0]['score']
        false_word_score = filled[1]['score']
    else:
        correct_word_score = filled[1][0]['score']
        false_word_score = filled[0][1]['score']
    return correct_word_score, false_word_score


def get_pythia_scores(model_name, model_revision, text, correct_word, false_word):
  """
  :param model_name: name including no. of params, e.g., "EleutherAI/pythia-70m-deduped"
  :param model_revision: num of training steps,e.g., "step3000"
  :return: scores of the correct next word and the false word, as floats
  """
  model = GPTNeoXForCausalLM.from_pretrained(
      model_name,
      revision=model_revision
      # cache_dir=f"./{model_name}/{model_revision}",
    )
  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      revision=model_revision
      # cache_dir=f"./{model_name}/{model_revision}",
    )

  inputs = tokenizer(text, return_tensors="pt")
  print(inputs)
  scores = model(**inputs).logits[0][-1]
  probs = scores.softmax(dim=0)

  i = 0
  correct_word_token = tokenizer.encode(correct_word)[i]
  false_word_token = tokenizer.encode(false_word)[i]
  print(tokenizer.decode(correct_word_token), tokenizer.decode(false_word_token))
  while correct_word_token == false_word_token:
      print(correct_word_token, false_word_token)
      i += 1
      if len(tokenizer.encode(correct_word)) - 1 < i or len(tokenizer.encode(false_word)) - 1 < i:
          print("one of the answers is prefix of the other")
          return
      correct_word_token = tokenizer.encode(correct_word)[i]
      false_word_token = tokenizer.encode(false_word)[i]


  correct_word_score = probs[correct_word_token].item()
  false_word_score = probs[false_word_token].item()
  return correct_word_score, false_word_score


if __name__ == '__main__':
    # get_bert_scores('google/multiberts-seed_4-step_20k', "The plumber that called Joy drove a grey truck. Therefore, [MASK] [MASK] drove a grey truck.",
    #                 "plumber", "John", 2)
    print(get_pythia_scores("EleutherAI/pythia-70m-deduped","step100000", "Once upon a time in a","teacher", "teachernet" ))
    # print(get_pythia_scores("EleutherAI/pythia-70m-deduped", "step3000", "Jfsdsff ", "model", "robot"))