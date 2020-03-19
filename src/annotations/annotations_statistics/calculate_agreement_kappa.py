from sklearn.metrics import cohen_kappa_score

from config import data_dir, DISORDER, CHEMICAL_OR_DRUG
import json
import os
import pandas as pd

from high_recall_matcher_posts_level import words_similarity
SIMILARITY_THRESHOLD = 0.85

labels_dir = data_dir + r'manual_labeled_v2\doccano'
measuring_kappa_dir = labels_dir + os.sep + "csv_files" + os.sep + "measuring_kappa"

label_numbers_for_each_community = {"diabetes": {DISORDER: 9, CHEMICAL_OR_DRUG: 10},
                                    "sclerosis": {DISORDER: 7, CHEMICAL_OR_DRUG: 8},
                                    "depression": {DISORDER: 11, CHEMICAL_OR_DRUG: 12}}


def w_is_similar_to_word_in_lst(w, lst):
    for w2 in lst:
        if words_similarity(w, w2) > SIMILARITY_THRESHOLD:
            return True
    return False


def get_bio_tags(row, community, user_num):
    label_col = f"user_{str(user_num)}_labels"
    labels = row[label_col]
    text = row['text'].replace("\n", " ").replace("\r", " ").replace("?", " ").replace("(", " ").replace(")", " ").replace(",", " ")
    text_words = [w for w in text.split(" ") if w not in ['', ' ', ',', '?', '!', '\n', '/']]
    assert all(x in label_numbers_for_each_community[community].values() for x in [m['label'] for m in labels])

    all_disorder_tokens = [m['term'].strip() for m in labels if m['label'] == label_numbers_for_each_community[community][DISORDER]]
    all_chemical_tokens = [m['term'].strip() for m in labels if m['label'] == label_numbers_for_each_community[community][CHEMICAL_OR_DRUG]]

    all_chemicals_token_single_words, all_disorder_token_single_words = get_disorders_chemicals_token_single_words(
        all_chemical_tokens, all_disorder_tokens)

    text_tags = []

    for idx, w in enumerate(text_words):
        w_tag = get_w_tag(all_disorder_tokens, all_chemical_tokens, w)
        text_tags.append(w_tag)

    words_and_tags = list(zip(text_words, text_tags))

    fix_words_and_tags(all_chemicals_token_single_words, all_disorder_token_single_words, words_and_tags)

    return words_and_tags


def fix_words_and_tags(all_chemicals_token_single_words, all_disorder_token_single_words, words_and_tags):
    all_tags = [x[1] for x in words_and_tags]
    tags_with_d = [t for t in all_tags if 'D' in t]
    tags_with_c = [t for t in all_tags if 'C' in t]
    gap_disorders = gap_chemicals = 0
    if len(tags_with_d) > len(all_disorder_token_single_words):
        gap_disorders = len(tags_with_d) - len(all_disorder_token_single_words)
    if len(tags_with_c) > len(all_chemicals_token_single_words):
        gap_chemicals = len(tags_with_c) - len(all_chemicals_token_single_words)
    if gap_disorders == gap_chemicals == 0:
        return words_and_tags

    new_words_and_tags = []
    for w, t in words_and_tags:
        if gap_disorders > 0 and 'D' in t:
            gap_disorders -= 1
            t = 'O'
        if gap_chemicals > 0 and 'C' in t:
            gap_chemicals -= 1
            t = 'O'
        new_words_and_tags.append((w, t))

    return new_words_and_tags

def get_disorders_chemicals_token_single_words(all_chemical_tokens, all_disorder_tokens):
    all_disorder_token_single_words = []
    for d in all_disorder_tokens:
        all_disorder_token_single_words += [x for x in d.split(" ") if len(x) > 1]
    all_chemicals_token_single_words = []
    for c in all_chemical_tokens:
        all_chemicals_token_single_words += [x for x in c.split(" ") if len(x) > 1]
    return all_chemicals_token_single_words, all_disorder_token_single_words


def get_similar_term_from_lst(w, lst):
    for w2 in lst:
        w2_parts = w2.split(" ")
        for part_idx, w2_p in enumerate(w2_parts):
            if words_similarity(w, w2_p) > SIMILARITY_THRESHOLD:
                if part_idx == 0:
                    return w2_p, 'B'
                else:
                    return w2_p, 'I'
    # If no word in list, that have part relevant for w, no similar term from lst
    return None, None


def get_w_tag(all_disorders, all_chemicals, w):
    w2_p_disorder, loc_disorder = get_similar_term_from_lst(w, all_disorders)
    if w2_p_disorder and loc_disorder:
        if loc_disorder == 'I':
            w_tag = 'I-D'
        else:
            w_tag = 'B-D'
    else:
        w2_p_chemical, loc_chemical = get_similar_term_from_lst(w, all_chemicals)
        if w2_p_chemical and loc_chemical:
            if loc_chemical == 'I':
                w_tag = 'I-C'
            else:
                w_tag = 'B-C'
        else:
            w_tag = 'O'
    return w_tag


def calc_kappa(row):
    y1 = [x[1] for x in row['user_5_bio']]
    y2 = [x[1] for x in row['user_6_bio']]
    kappa = cohen_kappa_score(y1, y2)
    return kappa


def calculate_kappa_for_community(community):
    df = pd.read_csv(measuring_kappa_dir + os.sep + community + "_both_annotators.csv")

    for c in ['user_5_labels', 'user_6_labels']:
        df[c] = df[c].apply(lambda x: json.loads(x))

    df['user_5_bio'] = df.apply(lambda row: get_bio_tags(row, community, user_num=5), axis=1)
    df['user_6_bio'] = df.apply(lambda row: get_bio_tags(row, community, user_num=6), axis=1)

    df['kappa'] = df.apply(lambda row: calc_kappa(row), axis=1)

    kappa_mean = df['kappa'].mean()

    print(f"{community} kappa_mean is: {round(kappa_mean, 3)}")

def main():
    calculate_kappa_for_community('sclerosis')
    calculate_kappa_for_community('diabetes')
    calculate_kappa_for_community('depression')


if __name__ == '__main__':
    main()