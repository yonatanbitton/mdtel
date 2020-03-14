
import pandas as pd
import os
from config import data_dir

initialized_training_dataset_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
count_features_path = data_dir + r"contextual_relevance\count_features"
lm_features_path = data_dir + r"contextual_relevance\language_models\output"
relatedness_features_path = data_dir + r"contextual_relevance\relatedness\output"
yap_features_path = data_dir + r"contextual_relevance\yap_features"

extracted_training_dataset_dir = data_dir + r"contextual_relevance\extracted_training_dataset"


def handle_community(community):
    initialized_training_dataset = pd.read_csv(initialized_training_dataset_dir + os.sep + community + ".csv")
    yap_features = pd.read_csv(yap_features_path + os.sep + community + "_output.csv")
    count_features = pd.read_csv(count_features_path + os.sep + community + "_output.csv")
    lm_features = pd.read_csv(lm_features_path + os.sep + community + "_output.csv")
    relatedness_features = pd.read_csv(relatedness_features_path + os.sep + community + "_output.csv")

    column_lengths = len(initialized_training_dataset.columns), len(count_features.columns), len(lm_features.columns), len(relatedness_features.columns)

    joint_cols = ['cand_match', 'umls_match', 'file_name', 'tokenized_text', 'occurences_indexes', 'match_10_window',
                  'match_3_window', 'match_6_window', 'match_idx', 'row_idx']
    count_cols = ['match_count', 'match_freq']
    lm_cols = ['pred_10_window', 'pred_6_window', 'pred_3_window', 'pred_2_window']
    relatedness_cols = ['relatedness']
    yap_cols = ['dep_part', 'gen', 'pos', 'tense']
    all_final_cols = joint_cols + count_cols + lm_cols + relatedness_cols + yap_cols

    idx_col = 'match_occurence_idx_in_txt_words'
    print(f"Len before: {len(initialized_training_dataset), len(count_features), len(lm_features), len(relatedness_features), len(yap_features)}")

    all_rows = []
    for idx in range(len(initialized_training_dataset)):
        r1 = initialized_training_dataset.iloc[idx]
        r2 = count_features.iloc[idx]
        r3 = lm_features.iloc[idx]
        r4 = relatedness_features.iloc[idx]
        r5 = yap_features.iloc[idx]
        merged_d = {**dict(r1), **dict(r2), **dict(r3), **dict(r4), **dict(r5)}
        assert r1['cand_match'] == r2['cand_match'] == r3['cand_match'] == r4['cand_match'] == r5['cand_match'] and \
               r1['umls_match'] == r2['umls_match'] == r3['umls_match'] == r4['umls_match'] == r5['umls_match'] and \
               r1[idx_col] == r2[idx_col] == r3[idx_col] == r4[idx_col] == r5[idx_col] and \
               r1['row_idx'] == r2['row_idx'] == r3['row_idx'] == r4['row_idx'] == r5['row_idx']
        all_rows.append(merged_d)

    result = pd.DataFrame(all_rows, columns=all_final_cols)
    print(f"shapes before: {result.shape}, column lengths: {column_lengths}, joint cols: {len(joint_cols)}, new_res: {len(result.columns)}")
    result.to_csv(extracted_training_dataset_dir + os.sep + community + ".csv", index=False, encoding='utf-8-sig')
    print(f"merged written to file {extracted_training_dataset_dir + os.sep + community}")

def main():
    handle_community('sclerosis')
    handle_community('diabetes')
    handle_community('depression')

    print("Done")

if __name__ == '__main__':
    main()