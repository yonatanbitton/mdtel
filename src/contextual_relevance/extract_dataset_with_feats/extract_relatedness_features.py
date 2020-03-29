import os
import sys

import pandas as pd

from contextual_relevance.extract_dataset_with_feats.yap.yap_api import YapApi

module_path = os.path.abspath(os.path.join('..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, umls_similarity_path

input_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
calculated_relatedness_dir = data_dir + r"contextual_relevance\relatedness"

output_dir = data_dir + r"contextual_relevance\relatedness\output"

cuiless_dir = data_dir + r"manual_labeled_v2\items_not_in_umls"
cuiless_df = pd.read_excel(cuiless_dir + os.sep + "cuiless_terms.xlsx")
all_cuiless_terms = set(list(cuiless_df['STR']))
print("Added CUILESS terms")

yap = YapApi()
ip = '127.0.0.1:8000'

def handle_community(community):
    calculated_relatedness_dict = get_relatedness_dict(community)

    comm_df = pd.read_csv(input_dir + os.sep + community + ".csv")

    comm_no_relatedness = {'count': 0, 'terms': []}

    all_relatedness = []
    for row_idx, row in comm_df.iterrows():
        eng_match = row['match_eng']

        if eng_match in all_cuiless_terms:
            relatedness = 1
        elif eng_match not in calculated_relatedness_dict:
            splitted_term = eng_match.split("(")[0].strip()
            if splitted_term in calculated_relatedness_dict:
                relatedness = float(calculated_relatedness_dict[splitted_term])
            else:
                comm_no_relatedness['count'] += 1
                comm_no_relatedness['terms'].append(eng_match)
                relatedness = -1
        else:
            relatedness = float(calculated_relatedness_dict[eng_match])
        all_relatedness.append(relatedness)

    comm_df['relatedness'] = all_relatedness
    comm_no_relatedness['terms'] = set(comm_no_relatedness['terms'])
    if comm_no_relatedness['count'] > 0:
        prepare_file_to_calculate_relatedness_for_missing_terms(comm_no_relatedness, community)

    comm_df.to_csv(output_dir + os.sep + community + "_output.csv", index=False, encoding='utf-8-sig')


def prepare_file_to_calculate_relatedness_for_missing_terms(comm_no_relatedness, community):
    print(f"Comm: {community}, No relatedness count: {comm_no_relatedness['count']}, num_terms: {len(comm_no_relatedness['terms'])}")
    completions_p = umls_similarity_path + os.sep + 'completions' + os.sep
    fname = community + "_completions.txt"
    with open(completions_p + os.sep + fname, 'w') as f:
        for t in comm_no_relatedness['terms']:
            f.write(community + "<>" + t + "\n")
    output_fname = community + "_completions_output.txt"
    cmd = f'umls-similarity.pl -user=root -password=admin --infile="completions/{fname}" --measure=vector > completions/{output_fname}'
    print(cmd)


def get_relatedness_dict(community):
    with open(calculated_relatedness_dir + os.sep + community + "_relatedness.txt", encoding='utf-8') as f:
        lines = [x.rstrip('\n') for x in f.readlines()]
    calculated_relatedness_dict = {}
    for l in lines:
        relatedness = l.split("<>")[0]
        term = l.split("<>")[-1].split("(")[0]
        calculated_relatedness_dict[term] = relatedness
    return calculated_relatedness_dict


if __name__ == '__main__':
    print(f"Extracting relatedness features")

    handle_community("diabetes")
    handle_community("sclerosis")
    handle_community("depression")
    print("Relatedness extractor - Done.")
