import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsort_keygen
from plot_utils import get_plot_values


COLORS = ['green', 'purple', 'deepskyblue', 'orange', 'red', 'pink', "deeppink", "limegreen", "navy", "orangered"]
ALL_MODELS =  ["google/multiberts-seed_0", "google/multiberts-seed_3",
               "EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]

def plot_results(title, xlabel, ylabel, xticks, xaxis_col, yaxis_cols, legend_names, results_filename):
    fig, ax = plt.subplots()
    lines = []
    for i,yaxis_col in enumerate(yaxis_cols):
        style = "dashed" if legend_names[i].startswith("Human") else "solid"
        line = ax.plot(xaxis_col, yaxis_col, color=COLORS[i], label= legend_names[i], linestyle = style)
        lines.append(line)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(title, fontsize = 8)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(xticks, fontsize = 5)
    ax.set_ylabel(ylabel)
    plt.savefig(results_filename, dpi=300, format="png")


def plot_bert_average_acc():
    df_path = "../all_analysis.csv"
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
    df_path = "../all_analysis.csv"
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
        results_filename="../accuracy_plots/google_multiberts_accuracy_mask_1.png"
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
        results_filename="../accuracy_plots/google_multiberts_accuracy_mask_2.png"
    )

def plot_bert_average_acc_considering_num_of_masks():
    df_path = "../all_analysis_num_of_mask_vers.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    legend_names = ["google/multiberts-seed_0/1_num_of_masks", "google/multiberts-seed_3/1_num_of_masks",
                    "google/multiberts-seed_0/2_num_of_masks", "google/multiberts-seed_3/2_num_of_masks"]

    xcol_1, xticks_1, ycols_1 = get_plot_values(df, models, col_name="accuracy_1_num_mask")
    xcol_2, xticks_2, ycols_2 = get_plot_values(df, models, col_name="accuracy_2_num_mask")
    merged_ycols = ycols_1 + ycols_2

    plot_results(
        title="Multiberts - accuracy considering num of masks",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=xcol_1,
        yaxis_cols=merged_ycols,
        xticks=xticks_1,
        legend_names=legend_names,
        results_filename="../accuracy_plots/google_multiberts_accuracy_num_of_masks.png"
    )


def plot_pythia_average_acc():
    df_path = "../all_analysis.csv"
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
        results_filename="../accuracy_plots/pythia_accuracy.png"
    )

    # cols = ['model_name', 'model_step','false_mask1_avg', 'correct_mask1_avg', 'false_mask2_avg', 'correct_mask2_avg', 'accuracy_mask1', 'accuracy_mask2']

def plot_pythia_average_acc_considering_num_of_masks():
    df_path = "../all_analysis_num_of_mask_vers.csv"
    df = pd.read_csv(df_path)
    models= ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
    legend_names = ["EleutherAI/pythia-70m/1_num_of_masks", "EleutherAI/pythia-410m/1_num_of_masks",
                    "EleutherAI/pythia-2.8b/1_num_of_masks", "EleutherAI/pythia-70m/2_num_of_masks",
                    "EleutherAI/pythia-410m/2_num_of_masks", "EleutherAI/pythia-2.8b/2_num_of_masks"]

    xcol_1, xticks_1, ycols_1 = get_plot_values(df, models, col_name="accuracy_1_num_mask")
    xcol_2, xticks_2, ycols_2 = get_plot_values(df, models, col_name="accuracy_2_num_mask")
    merged_ycols = ycols_1 + ycols_2

    plot_results(
        title="Pythia - accuracy considering num of masks",
        xlabel="step",
        ylabel="accuracy",
        xaxis_col=xcol_1,
        yaxis_cols=merged_ycols,
        xticks=xticks_1,
        legend_names=legend_names,
        results_filename="../accuracy_plots/pythia_accuracy_num_of_masks.png"
    )


def apply_int_step_to_df(df_lst):
    for df in df_lst:
        df['model_step_int'] = df['model_step'].apply(lambda x: x[5:-1] if x.endswith('k') else x[4:]) # differs between pythia and bert

def truncate_df_to_model_and_sort(df,model):
    df = df[df["model_name"]== model]
    df.sort_values(by='model_step', inplace=True, key=natsort_keygen())
    return df


def plot_animate_results_specific_model():
    false_animate_df = pd.read_csv(r'analysis\animate_state\false_animate_analysis.csv')
    both_animate_df = pd.read_csv(r'analysis\animate_state\both_animate_analysis.csv')
    criteria_names = {'correct_mask1_avg': 'correct answer average probability',\
                      'false_mask1_avg': 'false answer average probability' ,'accuracy_mask1':'Accuracy'}
    apply_int_step_to_df([false_animate_df,both_animate_df])
    models = set(false_animate_df['model_name'])
    Human_baselines = [0.51, 0.77] # Both animated and False animated respectively
    for model in models:
        false_animated_df_model= truncate_df_to_model_and_sort(false_animate_df,model)
        both_animated_df_model= truncate_df_to_model_and_sort(both_animate_df,model)
        for criteria,name in criteria_names.items():
            legend_names=["false animated", "both animated"]
            ycols = [false_animated_df_model[criteria], both_animated_df_model[criteria]]
            xcols = false_animated_df_model["model_step_int"]
            if name == "Accuracy":
                ycols.append([Human_baselines[0] for i in range(len(xcols))])
                ycols.append([Human_baselines[1] for i in range(len(xcols))])
                legend_names.append("Human both animated")
                legend_names.append("Human false animated")                
            plot_results(
            title=f"Model {model}, Both Animated Vs False Animated- {name}",
            xlabel="step",
            ylabel=name,
            xaxis_col=xcols,
            yaxis_cols= ycols,
            xticks=false_animated_df_model["model_step"],
            legend_names=legend_names,
            results_filename=rf"analysis\animate_state\animated_graphs\{model}_{name}.png"
            )

def plot_animate_results_all_models_in_graph():
    animate_df = pd.read_csv(r'analysis\animate_state\false_animate_analysis.csv')
    inanimate_df = pd.read_csv(r'analysis\animate_state\both_animate_analysis.csv')
    criteria_names = {'correct_mask1_avg': 'correct answer average probability',\
                      'false_mask1_avg': 'false answer average probability' ,'accuracy_mask1':'Accuracy'}
    apply_int_step_to_df([animate_df,inanimate_df])
    model_names = ["Bert", "Pythia"]
    models_lst= [["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"],["google/multiberts-seed_0", "google/multiberts-seed_3"]]
    # models_lst= [["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"],["google/multiberts-seed_0", "google/multiberts-seed_3"]]
    Human_baselines = [0.51, 0.77] # Both animated and False animated respectively
    for models in models_lst:
        for criteria,name in criteria_names.items():
            legend_names = []
            ycols = []
            xcol1, xticks1, ycols1 = get_plot_values(inanimate_df, models, col_name=criteria)
            ycols.extend(ycols1)
            legend_names.extend([model+" Both animated" for model in models])
            xcol2, xticks2, ycols2 = get_plot_values(animate_df, models, col_name=criteria)
            ycols.extend(ycols2)
            legend_names.extend([model+" False animated" for model in models])
            if name == "Accuracy":
                ycols.append([Human_baselines[0] for i in range(len(xcol1))])
                ycols.append([Human_baselines[1] for i in range(len(xcol1))])
                legend_names.append("Human both animated")
                legend_names.append("Human false animated")                
            model_name = models[0].split('/')[1].split('-')[0].capitalize()
            plot_results(
            title=f"{model_name}, Both Animated Vs False Animated- {name}",
            xlabel="step",
            ylabel=name,
            xaxis_col=xcol1,
            yaxis_cols=ycols,
            xticks=xticks1,
            legend_names=legend_names,
            results_filename=rf"analysis\animate_state\animated_graphs\{model_name}_{name}.png"
            )

def plot_pythia_false_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models= ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="false_mask1_avg")
    plot_results(
        title="Pythia - False Scores Average Over Steps",
        xlabel="Step",
        ylabel="False Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/pythia_false_scores_avg.png"
    )

def plot_pythia_correct_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models= ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="correct_mask1_avg")
    plot_results(
        title="Pythia - Correct Scores Average Over Steps",
        xlabel="Step",
        ylabel="Correct Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/pythia_correct_scores_avg.png"
    )

def plot_bert_mask_1_false_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="false_mask1_avg")
    plot_results(
        title="Bert - Mask 1 False Scores Average Over Steps",
        xlabel="Step",
        ylabel="Mask 1 False Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/bert_mask_1_false_scores_avg.png"
    )

def plot_bert_mask_1_correct_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="correct_mask1_avg")
    plot_results(
        title="Bert - Mask 1 Correct Scores Average Over Steps",
        xlabel="Step",
        ylabel="Mask 1 Correct Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/bert_mask_1_correct_scores_avg.png"
    )

def plot_bert_mask_2_false_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="false_mask2_avg")
    plot_results(
        title="Bert - Mask 2 False Scores Average Over Steps",
        xlabel="Step",
        ylabel="Mask 2 False Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/bert_mask_2_false_scores_avg.png"
    )

def plot_bert_mask_2_correct_scores_avg():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    xcol, xticks, ycols = get_plot_values(df, models, col_name="correct_mask2_avg")
    plot_results(
        title="Bert - Mask 2 Correct Scores Average Over Steps",
        xlabel="Step",
        ylabel="Mask 2 Correct Scores Average",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../accuracy_plots/bert_mask_2_correct_scores_avg.png"
    )

def plot_pythia_correct_false_ratio():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models= ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
    xcol, xticks, ycols1 = get_plot_values(df, models, col_name="correct_mask1_avg")
    xcol1, xticks1, ycols2 = get_plot_values(df, models, col_name="false_mask1_avg")
    ycols = [ycols1[i]/ycols2[i] for i in range(len(ycols1))]

    plot_results(
        title="Pythia - Correct/False answer ratio",
        xlabel="Step",
        ylabel="Average Correct/False ratio",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../ratio_plots/pythia_avg_ratio.png"
    )

def plot_bert_correct_false_ratio():
    df_path = "../all_analysis.csv"
    df = pd.read_csv(df_path)
    models = ["google/multiberts-seed_0", "google/multiberts-seed_3"]
    xcol, xticks, ycols1 = get_plot_values(df, models, col_name="correct_mask1_avg")
    xcol1, xticks1, ycols2 = get_plot_values(df, models, col_name="false_mask1_avg")
    ycols = [ycols1[i]/ycols2[i] for i in range(len(ycols1))]

    plot_results(
        title="Bert - Correct/False answer ratio",
        xlabel="Step",
        ylabel="Average Correct/False ratio",
        xaxis_col=xcol,
        yaxis_cols=ycols,
        xticks=xticks,
        legend_names=models,
        results_filename="../ratio_plots/bert_avg_ratio.png"
    )


if __name__ == '__main__':
    # plot_bert_average_acc()
    # plot_multiple_bert_average_acc()
    # plot_animate_results_all_models_in_graph()
    plot_animate_results_specific_model()
    # plot_pythia_average_acc()
    # plot_pythia_false_scores_avg()
    # plot_pythia_correct_scores_avg()
    # plot_bert_mask_1_false_scores_avg()
    # plot_bert_mask_1_correct_scores_avg()
    # plot_bert_mask_2_false_scores_avg()
    # plot_bert_mask_2_correct_scores_avg()
    # plot_bert_average_acc_considering_num_of_masks()
    # plot_pythia_average_acc_considering_num_of_masks()
    # plot_pythia_correct_false_ratio()
    # plot_bert_correct_false_ratio()
    # plot_results(None, "my title", "x title", "y title", [3,4,10], [[4,5,7], [2,3,6], [1,1,1]], ["name1", "name2", "name3"], "trial3.png")


