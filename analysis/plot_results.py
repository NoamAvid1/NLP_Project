import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsort_keygen

colors = ['green', 'purple', 'deepskyblue', 'orange', 'red', 'pink']


def plot_results(df, title, xlabel, ylabel, xticks, xaxis_col, yaxis_cols, legend_names, results_filename):
    fig, ax = plt.subplots()
    lines = []
    for i,yaxis_col in enumerate(yaxis_cols):
        line = ax.plot(xaxis_col, yaxis_col, color=colors[i])
        lines.append(line)
    ax.legend(lines, legend_names)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(ylabel)
    plt.savefig(results_filename, dpi=300, format="png")
    plt.show()


def plot_bert_average_acc():
    df_path = "../results_analysis/all_analysis.csv"
    df = pd.read_csv(df_path)
    df = df[df["model_name"] == "google_multiberts-seed_3"]  # filter only bert models seed 0
    df.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    df['model_step_int'] = df['model_step'].apply(lambda x: int(x[5:-1]))
    plot_results(
        df=df,
        title="Mask 1 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xticks=df['model_step'],
        xaxis_col=df["model_step_int"],
        yaxis_cols=[df["accuracy_mask1"]],
        legend_names=["google_multiberts-seed_3"],
        results_filename="google_multiberts-seed_3_accuracy.png"
    )


def plot_multiple_bert_average_acc():
    df_path = "../results_analysis/all_analysis.csv"
    df = pd.read_csv(df_path)
    df['model_step_int'] = df['model_step'].apply(lambda x: int(x[5:-1]))
    df_seed0 = df[df["model_name"] == "google_multiberts-seed_0"]  # filter only bert models seed 0
    df_seed3 = df[df["model_name"] == "google_multiberts-seed_3"]  # filter only bert models seed 3
    df_seed0.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    df_seed3.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    plot_results(
        df=df,
        title="Mask 1 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=df_seed0["model_step_int"],
        yaxis_cols=[df_seed0["accuracy_mask1"], df_seed3["accuracy_mask1"]],
        xticks=df_seed0["model_step"],
        legend_names=["google_multiberts-seed_0","google_multiberts-seed_3" ],
        results_filename="google_multiberts_accuracy.png"
    )

def plot_pythia_average_acc():
    df_path = "../analysis/all_analysis.csv"
    df = pd.read_csv(df_path)
    df['model_step_int'] = df['model_step'].apply(lambda x: int(x[5:-1]))
    df_seed0 = df[df["model_name"] == "google_multiberts-seed_0"]  # filter only bert models seed 0
    df_seed3 = df[df["model_name"] == "google_multiberts-seed_3"]  # filter only bert models seed 3
    df_seed0.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    df_seed3.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    plot_results(
        df=df,
        title="Mask 1 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=df_seed0["model_step_int"],
        yaxis_cols=[df_seed0["accuracy_mask1"], df_seed3["accuracy_mask1"]],
        xticks=df_seed0["model_step"],
        legend_names=["google_multiberts-seed_0", "google_multiberts-seed_3"],
        results_filename="google_multiberts_accuracy.png"
    )

    # cols = ['model_name', 'model_step','false_mask1_avg', 'correct_mask1_avg', 'false_mask2_avg', 'correct_mask2_avg', 'accuracy_mask1', 'accuracy_mask2']


if __name__ == '__main__':
    # plot_bert_average_acc()
    plot_multiple_bert_average_acc()
    # plot_results(None, "my title", "x title", "y title", [3,4,10], [[4,5,7], [2,3,6], [1,1,1]], ["name1", "name2", "name3"], "trial3.png")


