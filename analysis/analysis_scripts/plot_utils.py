import pandas as pd
from natsort import natsort_keygen


def get_model_values(df, model_name, col_name):
    df = df[df["model_name"] == model_name.replace("/", "_")]
    df['model_step_int'] = df['model_step'].apply(lambda x: x[5:-1] if x.endswith('k') else x[4:]) # differs between pythia and bert
    df.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    return df['model_step_int'], df['model_step'], df[col_name]


def get_plot_values(df, models, col_name="accuracy_mask1"):
    ycols = []
    xcol, xticks = [], []
    for model in models:
        xcol, xticks, ycol = get_model_values(df, model, col_name)
        ycols.append(ycol)
    return xcol, xticks, ycols