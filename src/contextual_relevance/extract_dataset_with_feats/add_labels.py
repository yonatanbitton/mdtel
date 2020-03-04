import difflib
import os
import pandas as pd
import json
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

labels_dir = data_dir + r"manual_labeled"
extracted_feats_dir = data_dir + r"contextual_relevance\extracted_training_dataset"

output_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"

if DEBUG:
    print(f"*** DEBUG MODE: Taking 100 rows only ***")


def handle_community(community):
    print(f"community: {community}")
    labeled_df = pd.read_excel(labels_dir + os.sep + community + ".xlsx")

    if DEBUG:
        labeled_df = labeled_df.head(101)
    labeled_df['manual_tag'] = labeled_df['manual_tag'].apply(lambda x: [] if str(x) == 'nan' else x.split(","))
    labeled_df['manual_tag'] = labeled_df['manual_tag'].apply(lambda lst: [x.strip() for x in lst])
    extract_feats_df = pd.read_excel(extracted_feats_dir + os.sep + community + ".xlsx")
    print(f"Original Shape: {extract_feats_df.shape}")

    all_labels = []
    for r_idx, row in extract_feats_df.iterrows():
        try:
            relevant_labeled_df = labeled_df.iloc[row['row_idx']]
        except Exception as ex:
            print(f"*** ERROR - Will it happen later ? *** {str(ex)}")
            extract_feats_df = extract_feats_df.head(r_idx)
            break

        assert relevant_labeled_df['tokenized_txt'] == row['tokenized_txt']

        if row['match'] in relevant_labeled_df['manual_tag']:
            all_labels.append(1)
        else:
            best_match, best_match_sim = get_best_match(relevant_labeled_df, row)
            if best_match:
                all_labels.append(1)
                # print(f'{r_idx}: {best_match}-{row["match"]}: {best_match_sim}')
            else:
                all_labels.append(0)

    print(f"Final Shape (breaked): {extract_feats_df.shape}")
    extract_feats_df['yi'] = all_labels
    extract_feats_df = extract_feats_df[['match', 'yi', 'match_10_window', 'match_3_window', 'match_6_window',
                                         'match_idx', 'row_idx', 'tokenized_txt', 'match_count', 'match_freq',
                                         'pred_10_window', 'pred_6_window', 'pred_3_window', 'pred_2_window',
                                         'relatedness']]
    print("Done")

    fpath = output_dir + os.sep + community + '.xlsx'
    print(f"Writing file at shape: {extract_feats_df.shape} to fpath: {fpath}")
    extract_feats_df.to_excel(fpath, index=False)


def get_best_match(relevant_labeled_df, row):
    best_match = None
    best_match_sim = 0
    for w in relevant_labeled_df['manual_tag']:
        sim = words_similarity(w, row['match'])
        if sim > 0.88:
            best_match_sim = sim
            best_match = w
    return best_match, best_match_sim


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


if __name__ == '__main__':
    handle_community('diabetes')
    handle_community('sclerosis')
    handle_community('depression')

    print("Done")