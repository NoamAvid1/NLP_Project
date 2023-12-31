import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline
import pandas as pd
import json
from tqdm import tqdm
import os
import argparse


def load_pythia_model_tokenizer(model_name: str, model_revision: str, num_gpus: int = 4, max_gpu_mem: str | None = None):
    """Load a model from Hugging Face."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_mem is None:
                kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = [11 for i in range(num_gpus)]
                kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
            else:
                kwargs["max_memory"] = {i: max_gpu_mem for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    # Load model
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=f"cache/{model_name}/{model_revision}",
        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=f"cache/{model_name}/{model_revision}",
        **kwargs
    )

    if (device == "cuda" and num_gpus == 1):
        model.to(device)

    return model, tokenizer

def get_correct_and_false_tokens(targets_scores, row):
    if row['num_of_masks'] == 1:
        false_tokens = [i for i in targets_scores if i['token_str'] == row['false_answer'].lower()]
        correct_mask_1_tokens = [i for i in targets_scores if i['token_str'] == row['correct_answer'].lower()]
        correct_mask_2_tokens = None

    else:
        false_tokens = [i for i in targets_scores[0] if i['token_str'] == row['false_answer'].lower()]
        correct_mask_1_tokens = [i for i in targets_scores[0] if i['token_str'] == "the"]
        correct_mask_2_tokens = [i for i in targets_scores[1] if i['token_str'] == row['correct_answer'].lower()]
    return false_tokens, correct_mask_1_tokens, correct_mask_2_tokens


def get_bert_question(row: pd.Series):
    bert_question = row['model_question'] + "[MASK]"
    if row['num_of_masks'] == 2:
        bert_question += " [MASK]"
    return bert_question


def get_df_bert_scores(model_name, model_revision, row):
    """
    :param model_seed_and_step: for example- 'google/multiberts-seed_4-step_20k'
    :param text: masked sentence that we want to test
    :return: scores of the correct masked word and the false word, as floats
    """
    bert_question = get_bert_question(row)
    model = AutoModelForMaskedLM.from_pretrained(f"{model_name}-{model_revision}", cache_dir=f"cache/{model_name}/{model_revision}")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}-{model_revision}", cache_dir=f"cache/{model_name}/{model_revision}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, device = device)
    if row['num_of_masks'] == 1:
        targets_scores = unmasker(bert_question, targets=[row['false_answer'], row['correct_answer']])
    else:
        targets_scores = unmasker(bert_question,
                                  targets=[row['false_answer'].lower(), "the", row['correct_answer'].lower()])
    false_tokens, correct_tokens, correct_tokens2 = get_correct_and_false_tokens(targets_scores, row)
    # check that the correct and false words are in the model vocabulary:
    if len(correct_tokens) == 0 or len(false_tokens) == 0:
        return pd.NA, pd.NA, pd.NA
    if type(correct_tokens2) == list and len(correct_tokens2) == 0:
        return pd.NA, pd.NA, pd.NA
    false_word_score = false_tokens[0]['score']
    correct_mask_1_score = correct_tokens[0]['score']
    correct_mask_2_score = correct_tokens2[0]['score'] if row['num_of_masks'] == 2 else pd.NA
    return false_word_score, correct_mask_1_score, correct_mask_2_score


def get_df_pythia_scores(model_name, model_revision, row):
    # model, tokenizer = load_pythia_model_tokenizer(model_name, model_revision)
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=f"cache/{model_name}/{model_revision}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=f"cache/{model_name}/{model_revision}"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = tokenizer(row['model_question'], return_tensors="pt")
    inputs = inputs.to(device)
    scores = model(**inputs).logits[0][-1]
    probs = scores.softmax(dim=0)
    if row["num_of_masks"] == 2:
        correct_word = "the"
    else:
        correct_word = row['correct_answer']
    # taking the first different token
    i = 0
    correct_word_token = tokenizer.encode(correct_word)[i]
    false_word_token = tokenizer.encode(row['false_answer'])[i]
    while correct_word_token == false_word_token:
        i += 1
        if len(tokenizer.encode(correct_word)) - 1 < i or len(tokenizer.encode(row['false_answer'])) - 1 < i:
            return pd.NA, pd.NA
        correct_word_token = tokenizer.encode(correct_word)[i]
        false_word_token = tokenizer.encode(row['false_answer'])[i]

    false_word_score = probs[false_word_token].item()
    correct_word_score = probs[correct_word_token].item()
    return false_word_score, correct_word_score


def run_model_and_save(model_type, models_config, df):
    """
    :param model_type: Bert or Pythia
    :param: models_config: dictionary of all model version that we want to run
    :param: df: dataframe of preprocessed experiment data
    :return:
    """
    tqdm.pandas()
    for model in models_config["model_types"][model_type]['models']:
        for checkpoint in models_config["model_types"][model_type]['checkpoints']:
            print(f"running model {model} with checkpoint {checkpoint}")
            model_df = df.copy()
            if model_type == "bert":
                model_df[['false_answer_score', 'correct_mask_1_score', 'correct_mask_2_score']] = \
                    model_df.progress_apply(
                    lambda row: get_df_bert_scores(model, checkpoint, row), axis=1, result_type='expand')
            else:
                model_df[['false_answer_score', 'correct_mask_1_score']] = \
                    model_df.progress_apply(
                    lambda row: get_df_pythia_scores(model, checkpoint, row), axis=1, result_type='expand')
            results_dir = os.path.join("experiment_results", models_config["run_name"])
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            output_filename = f'{results_dir}/{model.replace("/", "_")}_{checkpoint}.csv'
            model_df.to_csv(output_filename, index=False)
            print(f"saved result to {output_filename}")

def parse_conf_file():
    parser.add_argument('--conf_file', type=str, help='model configuration file')
    args = parser.parse_args()
    return args.conf_file



if __name__ == '__main__':
    df = pd.read_csv('preprocessed_data/final_table/experiment_sentences_preprocessed - All merged.csv')
    parser = argparse.ArgumentParser()
    model_config_file = os.path.join('models_configs',parse_conf_file())
    models_config = {"model_types": []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} as device")
    with open(model_config_file) as f:
            models_config = json.load(f)
    for model_type in models_config["model_types"].keys():
        run_model_and_save(model_type, models_config, df)
