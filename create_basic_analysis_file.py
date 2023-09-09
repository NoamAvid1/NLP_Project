import pandas as pd
import argparse
import os


def get_average(df,cols):
    false_avg = df[cols[0]].mean()
    correct_mask_1_avg = df[cols[1]].mean()
    if cols[2] in df.columns:
        false_avg_mask_2_avg = df[cols[0]][df[cols[2]].notna()].mean()
        correct_mask_2_avg = df[cols[2]][df[cols[2]].notna()].mean()
    else:
        false_avg_mask_2_avg = pd.NA
        correct_mask_2_avg = pd.NA
    return {'false_mask1_avg': false_avg, 'correct_mask1_avg': correct_mask_1_avg, \
            'false_mask2_avg': false_avg_mask_2_avg, 'correct_mask2_avg':correct_mask_2_avg }

def get_accuracy(df):
    df['is_greater_mask_1'] = df['correct_mask_1_score'] > df['false_answer_score']
    accuracy_mask_1 = df['is_greater_mask_1'].mean()
    accuracy_mask_2 = pd.NA
    if 'correct_mask_2_score' in df.columns:
         df['is_greater_mask_2'] =  df['correct_mask_2_score'] > df['false_answer_score']
         accuracy_mask_2 = df['is_greater_mask_2'][df['correct_mask_2_score'].notna()].mean()
    return {'accuracy_mask1': accuracy_mask_1, 'accuracy_mask2': accuracy_mask_2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir",type= str, help= 'directory of result files' )
    parser.add_argument("--out",type= str, help= 'file path to save output analysis')
    return parser.parse_args()   


if __name__ == '__main__':
    args = parse_args()
    parser = argparse.ArgumentParser()
    dir = args.res_dir
    files = os.listdir(dir)
    new_df = pd.DataFrame(columns=['model_name', 'model_step','false_mask1_avg', 'correct_mask1_avg', 'false_mask2_avg', 'correct_mask2_avg', 'accuracy_mask1', 'accuracy_mask2'])
    for file in files: 
        if file.startswith('google'):
            split_arg = 2
        else:
            split_arg = 1
        model_values =  {'model_name': '_'.join(file.split('_')[:-split_arg]),'model_step': '_'.join(file.split('_')[-split_arg:])[:-4]}
        file_path = os.path.join(dir,file)
        df = pd.read_csv(file_path)
        average_values = get_average(df,['false_answer_score','correct_mask_1_score','correct_mask_2_score'])
        accuracy_values = get_accuracy(df)
        model_values.update(average_values)
        model_values.update(accuracy_values)

        new_df = pd.concat([new_df,pd.DataFrame([model_values])],ignore_index=True)
    new_df.to_csv(args.out, index=False)
