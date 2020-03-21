import difflib
import json
import os
import sys

import pandas as pd

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, FINAL_LABELS_COL

labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'
extracted_feats_dir = data_dir + r"contextual_relevance\extracted_training_dataset"

output_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"

SIMILARITY_THRESHOLD = 0.85

def handle_community(community):
    # print(f"community: {community}")

    extract_feats_df = pd.read_csv(extracted_feats_dir + os.sep + community + ".csv")
    extract_feats_df['all_match_occ'] = extract_feats_df['all_match_occ'].apply(lambda x: json.loads(x) if str(x) != 'nan' else [])
    extract_feats_df['curr_occurence_offset'] = extract_feats_df['curr_occurence_offset'].apply(lambda x: int(x) if str(x).isdigit() else x)

    # print(f"Original Shape: {extract_feats_df.shape}")

    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df['merged_inner_and_outer'] = labels_df['merged_inner_and_outer'].apply(json.loads)

    relevant_feats_df = extract_feats_df[extract_feats_df['file_name'].isin(list(labels_df['file_name'].values))]
    # print(f"extract_feats_df shape: {extract_feats_df.shape}, relevant_feats_df.shape: {relevant_feats_df.shape}")


    all_labels = []
    for r_idx, row in relevant_feats_df.iterrows():

        label_row = labels_df[labels_df['file_name'] == row['file_name']].iloc[0]

        assert label_row['tokenized_text'] == row['tokenized_text']

        if row['curr_occurence_offset'] == 'lemma' or row['curr_occurence_offset'] is None or str(row['curr_occurence_offset']) == 'nan':
            yi = get_yi_for_lemma(extract_feats_df, label_row, r_idx, row)
        else:
            yi = get_yi_for_cand_match(label_row, row)
        all_labels.append(yi)

    # print(f"Final Shape (breaked): {relevant_feats_df.shape}")
    curr_cols = relevant_feats_df.columns
    relevant_feats_df['yi'] = all_labels
    cols_order = curr_cols.insert(1, 'yi')
    relevant_feats_df = relevant_feats_df[cols_order]
    # print("Done")

    fpath = output_dir + os.sep + community + '.csv'
    # print(f"Writing file at shape: {relevant_feats_df.shape} to fpath: {fpath}")
    relevant_feats_df.to_csv(fpath, index=False, encoding='utf-8-sig')


def get_yi_for_cand_match(label_row, row):
    row_start_offset = row['curr_occurence_offset']
    row_end_offset = row['curr_occurence_offset'] + len(row['cand_match'])
    found_label_for_row = False
    for ann in label_row[FINAL_LABELS_COL]:
        if abs(ann['start_offset'] - row_start_offset) <= 2 and abs(ann['end_offset'] - row_end_offset) <= 2:
            if words_similarity(ann['term'], row['cand_match']) > SIMILARITY_THRESHOLD:
                found_label_for_row = True
    yi = 1 if found_label_for_row else 0
    return yi


def get_yi_for_lemma(extract_feats_df, label_row, r_idx, row):
    if row['cand_match'] in label_row[FINAL_LABELS_COL] or row['umls_match'] in label_row[FINAL_LABELS_COL]:
        yi = 1
    else:
        best_match, best_match_sim = get_best_match(label_row, row, FINAL_LABELS_COL)
        if best_match:
            extract_feats_df.at[r_idx, 'match'] = best_match
            yi = 1
        else:
            yi = 0
    return yi


def get_best_match(relevant_labeled_df, row, final_label_col):
    best_match = None
    best_match_sim = 0
    for match_data in relevant_labeled_df[final_label_col]:
        term = match_data['term']
        sim1 = words_similarity(term, row['cand_match'])
        sim2 = words_similarity(term, row['umls_match'])
        higher_sim = max(sim1, sim2)
        if higher_sim > SIMILARITY_THRESHOLD:
            best_match_sim = higher_sim
            best_match = term
    return best_match, best_match_sim


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


if __name__ == '__main__':
    handle_community('depression')
    handle_community('sclerosis')
    handle_community('diabetes')

    print("Adding labels - Done.")
