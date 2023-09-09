import pandas as pd
import argparse
import os

def apply_average(df,cols):
    # df[col for col in cols] = df.apply(lambda row: func(row), axis=1, result_type='expand')
    false_avg = df[cols[0]].mean()
    correct_mask_1_avg = df[cols[1]].mean()
    df.loc[len(df),cols[0]] = false_avg
    df.loc[len(df[cols[1]]),cols[1]] = correct_mask_1_avg
    if len(cols) > 2:
        false_avg_mask_2_avg = df[cols[0][:-1]][df[cols[2]].notna()].mean()
        correct_mask_2_avg = df[cols[2]][df[cols[2]].notna()].mean()
        df = pd.concat([df,{cols[0]:false_avg_mask_2_avg, cols[2]: correct_mask_2_avg}])
    return df


if __name__ == '__main__':
    dir = 'experiment_results'
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir,file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            new_df = apply_average(df,['false_answer_score','correct_answer_score'])
            new_df.to_csv(file_path, index=False)
