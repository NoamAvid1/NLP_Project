import os
import json

def remove_duplications_in_list(lst):
    sentences_set = set()
    ret_lst = []
    for element in lst:
        if element['sentence'] not in sentences_set:
            sentences_set.add(element['sentence'])
            ret_lst.append(element['sentence'])
    return ret_lst


if __name__ == '__main__':
    dir = '../data_exp'
    sub_dirs = [sub_dir for sub_dir in os.listdir(dir) if os.path.isdir(os.path.join(dir, sub_dir))]
    sentences = []
    for sub_dir in sub_dirs:
        file = os.path.join(dir,sub_dir,f'experiment_{sub_dir}.jsonl')
        with open(file,"r") as json_file:
            data_list = [json.loads(line.strip()) for line in json_file]
            no_dups_list = remove_duplications_in_list(data_list)
            sentences.extend(no_dups_list)
    with open('experiment_sentences.csv', 'w') as csv_file:
            for sentence in sentences:
                csv_file.write(f"{sentence}\n")