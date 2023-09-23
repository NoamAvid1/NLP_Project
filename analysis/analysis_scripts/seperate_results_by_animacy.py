import pandas as pd
import os


if __name__ == '__main__':
    animacy_csv_path = r'analysis\animate_state\experiment_sentences_preprocessed_animate_state.csv'
    results_dir = r'run_results\all_run'
    files = os.listdir(results_dir)
    animacy_class_df = pd.read_csv(animacy_csv_path)
    for file in files:
        df = pd.read_csv(os.path.join(results_dir,file))
        df['animacy_state'] = animacy_class_df['animacy_state']
        inanimate_df = df[df['animacy_state'] == 'both_animate']
        animate_df =  df[df['animacy_state'] == 'false_animate']
        inanimate_df.to_csv(os.path.join(r'analysis\animate_state\both_animate_results',file))
        animate_df.to_csv(os.path.join(r'analysis\animate_state\false_animate_results',file))