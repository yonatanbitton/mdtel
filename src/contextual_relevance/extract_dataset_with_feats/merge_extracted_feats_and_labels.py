import json

import pandas as pd
import os

from sklearn.model_selection import train_test_split

from config import data_dir

initialized_training_dataset_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
count_features_path = data_dir + r"contextual_relevance\count_features"
lm_features_path = data_dir + r"contextual_relevance\language_models\output"
relatedness_features_path = data_dir + r"contextual_relevance\relatedness\output"
yap_features_path = data_dir + r"contextual_relevance\yap_features"
labels_path = data_dir + r"contextual_relevance\labels"

extracted_training_dataset_dir = data_dir + r"contextual_relevance\extracted_training_dataset"


def handle_community(community):
    initialized_training_dataset = pd.read_csv(initialized_training_dataset_dir + os.sep + community + ".csv")
    initialized_training_dataset['txt_words'] = initialized_training_dataset['txt_words'].apply(json.loads)
    yap_features = pd.read_csv(yap_features_path + os.sep + community + "_output.csv")
    count_features = pd.read_csv(count_features_path + os.sep + community + "_output.csv")
    lm_features = pd.read_csv(lm_features_path + os.sep + community + "_output.csv")
    relatedness_features = pd.read_csv(relatedness_features_path + os.sep + community + "_output.csv")
    labels_df = pd.read_csv(labels_path + os.sep + community + "_labels.csv")

    joint_cols = list(initialized_training_dataset.columns)
    count_cols = ['match_count', 'match_freq']
    lm_cols = ['pred_10_window', 'pred_6_window', 'pred_3_window', 'pred_2_window']
    relatedness_cols = ['relatedness']
    yap_cols = ['dep_part', 'gen', 'pos', 'tense']
    label_col = ['yi']
    all_final_cols = joint_cols + label_col + count_cols + lm_cols + relatedness_cols + yap_cols

    idx_col = 'curr_occurence_offset'

    all_rows = []
    for idx in range(len(initialized_training_dataset)):
        r1 = initialized_training_dataset.iloc[idx]
        r2 = count_features.iloc[idx]
        r3 = lm_features.iloc[idx]
        r4 = relatedness_features.iloc[idx]
        r5 = yap_features.iloc[idx]
        r6 = labels_df.iloc[idx]
        remove_redundant_cols(r2, r3, r4, r5, r6)
        merged_d = {**dict(r1), **dict(r2), **dict(r3), **dict(r4), **dict(r5), **dict(r6)}
        for c in ['cand_match', 'umls_match', idx_col, 'row_idx']:
            assert r1[c] == r2[c] == r3[c] == r4[c] == r5[c] == r6[c]
        all_rows.append(merged_d)

    print(all_final_cols)
    result = pd.DataFrame(all_rows, columns=all_final_cols)

    test_size = 0.25

    all_filenames = result['file_name'].values
    all_filenames_nodups = list(set(all_filenames))
    filenames_train, filenames_test = train_test_split(all_filenames_nodups, test_size=test_size, shuffle=False)

    result_train = result[result['file_name'].isin(filenames_train)]
    result_test = result[result['file_name'].isin(filenames_test)]

    result_train.to_csv(extracted_training_dataset_dir + os.sep + "train" + os.sep + community + ".csv", index=False, encoding='utf-8-sig')
    result_test.to_csv(extracted_training_dataset_dir + os.sep + "test" + os.sep + community + ".csv", index=False, encoding='utf-8-sig')


def remove_redundant_cols(r2, r3, r4, r5, r6):
    for r in [r2, r3, r4, r5, r6]:
        for c in ['all_match_occ', 'txt_words', 'occurences_indexes', 'match']:
            if c in r:
                del r[c]


def main():
    handle_community('sclerosis')
    handle_community('diabetes')
    handle_community('depression')

    print("Merge extracted feats - Done.")

if __name__ == '__main__':
    main()