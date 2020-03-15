import difflib
import os
import pandas as pd
import json
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG, FINAL_LABELS_COL

# labels_dir = data_dir + r"manual_labeled"
labels_dir = data_dir + r'manual_labeled_v2\doccano'
extracted_feats_dir = data_dir + r"contextual_relevance\extracted_training_dataset"
posts_dir = data_dir + r"high_recall_matcher\posts\lemlda"

output_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"
labels_df_output_dir = data_dir + r'manual_labeled_v2\labels_dataframes'
annotations_length_per_user = {'user_5_labels': {'sclerosis': 407, 'diabetes': 516, 'depression': 376}}

SIMILARITY_THRESHOLD = 0.85

if DEBUG:
    print(f"*** DEBUG MODE: Taking 100 rows only ***")

def handle_community(community):
    print(f"community: {community}")
    with open(labels_dir + os.sep + community + "_export.json", encoding='utf-8') as f:
        comm_lines = [json.loads(x) for x in f.readlines()]
        comm_lines = comm_lines[:annotations_length_per_user[FINAL_LABELS_COL][community]]

    extract_feats_df = pd.read_csv(extracted_feats_dir + os.sep + community + ".csv")
    # for c in ['occurences_indexes_in_txt_words', 'txt_words', 'all_match_occ']:
    #     extract_feats_df[c] = extract_feats_df[c].apply(lambda x: json.loads(x) if str(x) != 'nan' else [])
    extract_feats_df['all_match_occ'] = extract_feats_df['all_match_occ'].apply(lambda x: json.loads(x) if str(x) != 'nan' else [])
    extract_feats_df['curr_occurence_offset'] = extract_feats_df['curr_occurence_offset'].apply(lambda x: int(x) if str(x).isdigit() else x)

    print(f"Original Shape: {extract_feats_df.shape}")

    posts_df = pd.read_excel(posts_dir + os.sep + community + "_posts.xlsx")

    labels_df = get_data_from_annotation_file(comm_lines, posts_df)
    labels_df = labels_df[labels_df[FINAL_LABELS_COL].apply(lambda x: len(x) > 0)]

    relevant_feats_df = extract_feats_df[extract_feats_df['file_name'].isin(list(labels_df['file_name'].values))]
    print(f"extract_feats_df shape: {extract_feats_df.shape}, relevant_feats_df.shape: {relevant_feats_df.shape}")


    all_labels = []
    for r_idx, row in relevant_feats_df.iterrows():

        label_row = labels_df[labels_df['file_name'] == row['file_name']].iloc[0]

        assert label_row['tokenized_text'] == row['tokenized_text']

        if row['curr_occurence_offset'] == 'lemma':
            yi = get_yi_for_lemma(extract_feats_df, label_row, r_idx, row)
        else:
            yi = get_yi_for_cand_match(label_row, row)
        all_labels.append(yi)

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


def get_representative_of_similar_term(t, terms_representatives):
    for w in terms_representatives:
        if words_similarity(t, w) > SIMILARITY_THRESHOLD:
            return w
    return None


def get_data_from_annotation_file(comm_lines, posts_df):
    all_line_texts = []
    all_line_tokenized_texts = []
    all_relevant_filenames = []
    all_user_5_labels = []
    all_user_6_labels = []
    all_line_txt_words = []
    for line_idx, line in enumerate(comm_lines):
        additional_post_data = posts_df[posts_df['post_txt'].apply(lambda x: line['text'].strip() == x.strip())].iloc[0]
        relevant_filename = additional_post_data['file_name']
        all_relevant_filenames.append(relevant_filename)
        tokenized_text_no_dots = additional_post_data['tokenized_text'].replace(".", " ") if str(additional_post_data['tokenized_text']) != 'nan' else 'nan'
        all_line_tokenized_texts.append(tokenized_text_no_dots)

        all_line_texts.append(line['text'])
        txt_words = line['text'].split(" ")
        all_line_txt_words.append(txt_words)

        line_annotations = line['annotations']
        if line_annotations == []:
            all_user_5_labels.append(line_annotations)
            all_user_6_labels.append(line_annotations)
        else:
            user_5_labels = []
            user_6_labels = []
            for ann in line_annotations:
                annotated_term = line['text'][ann['start_offset']:ann['end_offset']]
                term_d = {'term': annotated_term, 'start_offset': ann['start_offset'],
                          'end_offset': ann['end_offset'], 'label': ann['label']}
                if ann['user'] == 5:
                    user_5_labels.append(term_d)
                elif ann['user'] == 6:
                    user_6_labels.append(term_d)
                else:
                    print(f"Unknown annotation! {ann}")
            all_user_5_labels.append(user_5_labels)
            all_user_6_labels.append(user_6_labels)
    labels_df = pd.DataFrame()
    labels_df['text'] = all_line_texts
    labels_df['tokenized_text'] = all_line_tokenized_texts
    labels_df['txt_words'] = all_line_txt_words
    labels_df['file_name'] = all_relevant_filenames
    labels_df['user_5_labels'] = all_user_5_labels
    labels_df['user_6_labels'] = all_user_6_labels

    print(f"len(all_relevant_filenames): {len(all_relevant_filenames)}, "
          f"len(set(all_relevant_filenames)): {len(set(all_relevant_filenames))}")
    return labels_df


def get_suit_words(annotated_term, txt_words):
    suit_words = []
    for w in txt_words:
        sim = words_similarity(w, annotated_term)
        if sim > SIMILARITY_THRESHOLD:
            suit_words.append(w)
    return suit_words

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
    handle_community('sclerosis')
    handle_community('diabetes')
    handle_community('depression')

    print("Done")