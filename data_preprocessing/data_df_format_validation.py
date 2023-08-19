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



def main():
    df_path = "../preprocessed_data/experiment_sentences_preprocessed - Noam.csv"
    df = pd.read_csv(df_path)
    df['model_question'] = df.apply(lambda row: validate_original_sentence_from_model_question(row), axis=1)
    df.to_csv("../preprocessed_data/experiment_sentences_preprocessed - Noam 19.8.csv")

if __name__ == '__main__':
    main()