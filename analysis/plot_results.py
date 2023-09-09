import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsort_keygen
from plot_utils import get_plot_values


colors = ['green', 'purple', 'deepskyblue', 'orange', 'red', 'pink']


def plot_results(title, xlabel, ylabel, xticks, xaxis_col, yaxis_cols, legend_names, results_filename):
    fig, ax = plt.subplots()
    lines = []
    for i,yaxis_col in enumerate(yaxis_cols):
        line = ax.plot(xaxis_col, yaxis_col, color=colors[i], label= legend_names[i])
        lines.append(line)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(title, fontsize = 8)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(xticks, fontsize = 5)
    ax.set_ylabel(ylabel)
    plt.savefig(results_filename, dpi=300, format="png")


def plot_bert_average_acc():
    df_path = "../analysis/all_analysis.csv"
    df = pd.read_csv(df_path)
    df = df[df["model_name"] == "google_multiberts-seed_3"]  # filter only bert models seed 0
    df.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    df['model_step_int'] = df['model_step'].apply(lambda x: int(x[5:-1]))
    plot_results(
        title="MultiBerts - Mask 1 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xticks=df['model_step'],
        xaxis_col=df["model_step_int"],
        yaxis_cols=[df["accuracy_mask1"]],
        legend_names=["google_multiberts-seed_3"],
        results_filename="google_multiberts-seed_3_accuracy.png"
    )


def plot_multiple_bert_average_acc():
    df_path = "all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    # plot for mask 1
    xcol, xticks, ycols = get_plot_values(df, models)
    plot_results(
        title="Multiberts - Mask 1 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="accuracy_plots/google_multiberts_accuracy_mask_1.png"
    )
    # plot for mask 2
    xcol, xticks, ycols = get_plot_values(df, models, "accuracy_mask2")
    plot_results(
        title="Multiberts - Mask 2 Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="accuracy_plots/google_multiberts_accuracy_mask_2.png"
    )

def plot_pythia_average_acc():
    df_path = "../analysis/all_analysis.csv"
    df = pd.read_csv(df_path)
    models= ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
    xcol, xticks, ycols = get_plot_values(df, models)
    plot_results(
        title="Pythia - Accuracy Over Steps",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="accuracy_plots/pythia_accuracy.png"
    )

    # cols = ['model_name', 'model_step','false_mask1_avg', 'correct_mask1_avg', 'false_mask2_avg', 'correct_mask2_avg', 'accuracy_mask1', 'accuracy_mask2']


def apply_int_step_to_df(df_lst):
    for df in df_lst:
        df['model_step_int'] = df['model_step'].apply(lambda x: x[5:-1] if x.endswith('k') else x[4:]) # differs between pythia and bert

def truncate_df_to_model_and_sort(df,model):
    df = df[df["model_name"]== model]
    df.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    return df


def plot_animate_results():
    animate_df = pd.read_csv(r'analysis\animate_state\animate_analysis.csv')
    inanimate_df = pd.read_csv(r'analysis\animate_state\inanimate_analysis.csv')
    criteria_names = {'correct_mask1_avg': 'correct answer average probability',\
                      'false_mask1_avg': 'false answer average probability' ,'accuracy_mask1':'Accuracy'}
    apply_int_step_to_df([animate_df,inanimate_df])
    models = set(animate_df['model_name'])
    for model in models:
        animated_df_model= truncate_df_to_model_and_sort(animate_df,model)
        inanimated_df_model= truncate_df_to_model_and_sort(inanimate_df,model)
        for criteria,name in criteria_names.items():
            plot_results(
            title=f"Model {model}, Animate Vs Inanimate {name}",
            xlabel="step",
            ylabel=name,
            xaxis_col=animated_df_model["model_step_int"],
            yaxis_cols=[animated_df_model[criteria], inanimated_df_model[criteria]],
            xticks=animated_df_model["model_step"],
            legend_names=["animated", "inanimated"],
            results_filename=rf"analysis\animate_state\animated_graphs\{model}_{name}.png"
            )





if __name__ == '__main__':
    # plot_bert_average_acc()
    plot_multiple_bert_average_acc()
    # plot_animate_results()
    plot_pythia_average_acc()
    # plot_results(None, "my title", "x title", "y title", [3,4,10], [[4,5,7], [2,3,6], [1,1,1]], ["name1", "name2", "name3"], "trial3.png")


