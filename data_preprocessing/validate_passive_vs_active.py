from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
import json
from tqdm import tqdm
import os

def get_scores_for_sentence_type(sentences_dict: dict, correct_word, false_word):
    unmasker = pipeline('fill-mask', model=f"google/multiberts-seed_0-step_2000k",
                        tokenizer="google/multiberts-seed_0-step_2000k")
    for sentence_name in sentences_dict.keys():
        sentence = sentences_dict[sentence_name]
        target_scores = unmasker(sentence, targets=[correct_word, false_word])
        false_tokens = [i for i in target_scores if i['token_str'] == false_word.lower()]
        correct_tokens = [i for i in target_scores if i['token_str'] == correct_word.lower()]
        print(
            f"{sentence_name}: correct score: {correct_tokens[0]['score']}, false score: {false_tokens[0]['score']}")

        # print(target_scores)


if __name__ == '__main__':
    # sentences_dict=  {
    #     "Passive": "The architect that liked the officer dominated the conversation while the game was on television. Therefore, the conversation was dominated by the [MASK]",
    #     "Active - mask in the middle": "The architect that liked the officer dominated the conversation while the game was on television. Therefore, the [MASK] dominated the conversation",
    #     "Active - mask at the end": "The architect that liked the officer dominated the conversation while the game was on television. Therefore, the person who dominated the conversation is the [MASK]"
    # }
    sentences_dict = {
        "s1" : "The camera that was in the French director's studio this week hit the expensive vase at the entrance. Therefore, the [MASK] hit the expensive vase."
    }
    get_scores_for_sentence_type(sentences_dict, "camera", "director")
