import pandas as pd


def validate_original_sentence_from_model_question(row: pd.Series):
    """
    Removes original sentence from model question and re-adds it from the info
    in the df
    """
    model_question = row["model_question"]
    if type(model_question) != str:
        return model_question
    if not "Therefore, " in model_question:
        print(f"model question didn't contain 'Therefore,': {model_question}")
        return model_question
    splitted = model_question.split('Therefore, ')
    question_only = splitted[1]
    return f"{row['Sentence']} Therefore, {question_only}"


def validate_model_question_suffix(row: pd.Series):
    """
    This function validates that each input for the model ends with " "
    and that the inputs are formatted correctly according to the number of masks.
    """
    model_question = row["model_question"]
    if type(model_question) != str:
        return model_question
    if not row['model_question'][-1] == " ":
        print(f"fixing row: {model_question}")
        model_question += " "
    if row['num_of_masks'] == 1 and not model_question.endswith("the "):
        print(f"\n\nerror in row {row[0]}:\nnum_of_masks={row['num_of_masks']}\n model question: {model_question}\nFix manually")
    if row['num_of_masks'] == 2 and row['model_question'].endswith("the "):
        print(f"\n\nerror in row {row[0]}:\nnum_of_masks={row['num_of_masks']}\n model question: {model_question}\nFix manually")
    return model_question

def main():
    df_path = "../preprocessed_data/experiment_sentences_preprocessed - All merged.csv"
    df = pd.read_csv(df_path)
    df['model_question'] = df.apply(lambda row: validate_original_sentence_from_model_question(row), axis=1, result_type='expand')
    df['model_question'] = df.apply(lambda row: validate_model_question_suffix(row), axis=1, result_type='expand')
    df.to_csv("../preprocessed_data/experiment_sentences_preprocessed - All merged.csv")

if __name__ == '__main__':
    main()