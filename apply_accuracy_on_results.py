import pandas as pd
import os


def apply_accuracy_on_df(df):
    df['is_greater_mask_1'] = df['correct_mask_1_score'] > df['false_answer_score']
    df.loc[len(df),'is_greater_mask_1'] = df['is_greater_mask_1'].mean()
    if 'correct_mask_2_score' in df.columns:
         df['is_greater_mask_2'] =  df['correct_mask_2_score'] > df['false_answer_score']
         df.loc[len(df)-1,'is_greater_mask_2'] = df['is_greater_mask_2'][df['correct_mask_2_score'].notna()].mean()
    return df


if __name__ == '__main__':
    dir = r'experiment_results\trial'
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir,file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            new_df = apply_accuracy_on_df(df)
            new_df.to_csv(file_path, index=False)
