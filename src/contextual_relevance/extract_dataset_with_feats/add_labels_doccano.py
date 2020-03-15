import difflib
import os
import pandas as pd
import json
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG, FINAL_LABELS_COL, SIMILARITY_THRESHOLD

# labels_dir = data_dir + r"manual_labeled"
labels_dir = data_dir + r'manual_labeled_v2\doccano'
extracted_feats_dir = data_dir + r"contextual_relevance\extracted_training_dataset"
posts_dir = data_dir + r"high_recall_matcher\posts\lemlda"

output_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"
labels_df_output_dir = data_dir + r'manual_labeled_v2\labels_dataframes'

annotations_length_per_user = {'user_5_labels': {'sclerosis': 407, 'diabetes': 516, 'depression': 376}}

if DEBUG:
    print(f"*** DEBUG MODE: Taking 100 rows only ***")

def handle_community(community):
    print(f"community: {community}")
    with open(labels_dir + os.sep + community + "_export.json", encoding='utf-8') as f:
        comm_lines = [json.loads(x) for x in f.readlines()]
        comm_lines = comm_lines[:annotations_length_per_user[FINAL_LABELS_COL][community]]

    extract_feats_df = pd.read_csv(extracted_feats_dir + os.sep + community + ".csv")
    print(f"Original Shape: {extract_feats_df.shape}")

    posts_df = pd.read_excel(posts_dir + os.sep + community + "_posts.xlsx")

    labels_df = get_data_from_annotation_file(comm_lines, posts_df)

    labels_df = labels_df[labels_df[FINAL_LABELS_COL].apply(lambda x: len(x) > 0)]

    relevant_feats_df = extract_feats_df[extract_feats_df['file_name'].isin(list(labels_df['file_name'].values))]
    print(f"extract_feats_df shape: {extract_feats_df.shape}, relevant_feats_df.shape: {relevant_feats_df.shape}")

    all_labels = []
    for r_idx, row in relevant_feats_df.iterrows():

        relevant_labeled_df = labels_df[labels_df['file_name'] == row['file_name']].iloc[0]

        assert relevant_labeled_df['tokenized_text'] == row['tokenized_text']

        if row['cand_match'] in relevant_labeled_df[FINAL_LABELS_COL] or row['umls_match'] in relevant_labeled_df[FINAL_LABELS_COL]:
            all_labels.append(1)
        else:
            best_match, best_match_sim = get_best_match(relevant_labeled_df, row, FINAL_LABELS_COL)
            if best_match:
                extract_feats_df.at[r_idx, 'match'] = best_match
                all_labels.append(1)
                # print(f'{r_idx}: {best_match}-{row["match"]}: {best_match_sim}')
            else:
                all_labels.append(0)


    print(f"Final Shape (breaked): {relevant_feats_df.shape}")
    curr_cols = relevant_feats_df.columns
    relevant_feats_df['yi'] = all_labels
    cols_order = curr_cols.insert(1, 'yi')
    relevant_feats_df = relevant_feats_df[cols_order]
    print("Done")

    fpath = output_dir + os.sep + community + '.csv'
    print(f"Writing file at shape: {relevant_feats_df.shape} to fpath: {fpath}")
    relevant_feats_df.to_csv(fpath, index=False, encoding='utf-8-sig')

    labels_df['user_5_labels'] = labels_df['user_5_labels'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    labels_df['user_6_labels'] = labels_df['user_6_labels'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    labels_df.to_csv(labels_df_output_dir + os.sep + community + "_labeled.csv", index=False, encoding='utf-8-sig')


def get_data_from_annotation_file(comm_lines, posts_df):
    all_line_texts = []
    all_line_tokenized_texts = []
    all_relevant_filenames = []
    all_user_5_labels = []
    all_user_6_labels = []
    for line_idx, line in enumerate(comm_lines):
        additional_post_data = posts_df[posts_df['post_txt'].apply(lambda x: line['text'].strip() == x.strip())].iloc[0]
        relevant_filename = additional_post_data['file_name']
        all_relevant_filenames.append(relevant_filename)
        tokenized_text_no_dots = additional_post_data['tokenized_text'].replace(".", " ") if str(additional_post_data['tokenized_text']) != 'nan' else 'nan'
        all_line_tokenized_texts.append(tokenized_text_no_dots)

        all_line_texts.append(line['text'])

        line_annotations = line['annotations']
        if line_annotations == []:
            all_user_5_labels.append(line_annotations)
            all_user_6_labels.append(line_annotations)
        else:
            user_5_labels = []
            user_6_labels = []
            for ann in line_annotations:
                annotated_term = line['text'][ann['start_offset']:ann['end_offset']]
                if ann['user'] == 5:
                    user_5_labels.append(annotated_term)
                elif ann['user'] == 6:
                    user_6_labels.append(annotated_term)
                else:
                    print(f"Unknown annotation! {ann}")
            all_user_5_labels.append(user_5_labels)
            all_user_6_labels.append(user_6_labels)
    labels_df = pd.DataFrame()
    labels_df['text'] = all_line_texts
    labels_df['tokenized_text'] = all_line_tokenized_texts

    labels_df['file_name'] = all_relevant_filenames
    labels_df['user_5_labels'] = all_user_5_labels
    labels_df['user_6_labels'] = all_user_6_labels

    print(f"len(all_relevant_filenames): {len(all_relevant_filenames)}, "
          f"len(set(all_relevant_filenames)): {len(set(all_relevant_filenames))}")
    return labels_df


def remove_bad_char_and_lower(w : str):
    if "'" in w:
        w = w.replace("'", "")
    if word_is_english(w):
        w = w.lower()
    return w


def word_is_english(word):
   for c in word:
      if 'a' <= c <= 'z' or 'A' <= c <= 'C':
         return True
   return False

def get_best_match(relevant_labeled_df, row, final_label_col):
    best_match = None
    best_match_sim = 0
    for w in relevant_labeled_df[final_label_col]:
        sim1 = words_similarity(w, row['cand_match'])
        sim2 = words_similarity(w, row['umls_match'])
        higher_sim = max(sim1, sim2)
        if higher_sim > SIMILARITY_THRESHOLD:
            best_match_sim = higher_sim
            best_match = w
    return best_match, best_match_sim


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


if __name__ == '__main__':
    handle_community('sclerosis')
    handle_community('diabetes')
    handle_community('depression')

    print("Done")