import json
import os
import sys
from collections import defaultdict, Counter, namedtuple
from copy import deepcopy

import pandas as pd

from utils import words_similarity, replace_puncs


module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('..', '..', '..', os.getcwd()))
print(f"In extract_labels, {module_path}")
sys.path.append(module_path)

from config import data_dir, FINAL_LABELS_COL, LOW_SIMILARITY_THRESHOLD

labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'
extracted_feats_dir = data_dir + r"contextual_relevance\initialized_training_dataset"

output_dir = data_dir + r"contextual_relevance\labels"

cuiless_dir = data_dir + r"manual_labeled_v2\items_not_in_umls"
cuiless_df = pd.read_csv(cuiless_dir + os.sep + 'cuiless_terms.csv')
all_cuiless_terms = list(cuiless_df['HEB'].values)

missed_labels_counter = Counter()
most_common_n = 50

Entity = namedtuple("T", "i s e")


def handle_community(community):
    print(f"community: {community}")

    extract_feats_df, labels_df, relevant_feats_df = prepare_high_recall_and_labels_dfs(community)

    all_labels, high_recall_matches_matched_for_file_name, labels_df, relevant_feats_df = get_labels_for_high_recall_matches(
        extract_feats_df, labels_df, relevant_feats_df)

    all_labeled_terms_without_matches, could_be_matched = iterate_labels_calculate_stats(
        high_recall_matches_matched_for_file_name, labels_df, relevant_feats_df, community)

    print_fns_stats(all_labeled_terms_without_matches, community, could_be_matched)

    output_data(all_labels, community, relevant_feats_df)




def prepare_high_recall_and_labels_dfs(community):
    extract_feats_df = pd.read_csv(extracted_feats_dir + os.sep + community + ".csv")
    extract_feats_df['all_match_occ'] = extract_feats_df['all_match_occ'].apply(
        lambda x: json.loads(x) if str(x) != 'nan' else [])
    extract_feats_df['curr_occurence_offset'] = extract_feats_df['curr_occurence_offset'].apply(
        lambda x: int(x) if str(x).isdigit() else x)
    labels_df, relevant_feats_df = get_labels_df(community, extract_feats_df)

    return extract_feats_df, labels_df, relevant_feats_df


def get_labels_df(community, extract_feats_df):
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(json.loads)
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(lambda lst: strip_terms(lst))
    return labels_df, extract_feats_df

def fix_tokenization_problems(row, tagger):
    bio_tags = row['bio_tags']
    tokenization_problems = bio_tags['tokenization_problems']
    if len(tokenization_problems) == 0:
        return bio_tags['words_and_tags']
    else:
        labels = row[FINAL_LABELS_COL]
        good_labels = [t for t in labels if t not in tokenization_problems]
        assert len(good_labels) == len(labels) - len(tokenization_problems)
        row[FINAL_LABELS_COL] = good_labels
        bio_tags_fixed = tagger.get_bio_tags(row, FINAL_LABELS_COL)
        fixed_tokenization_problems = bio_tags_fixed['tokenization_problems']
        assert len(fixed_tokenization_problems) == 0
        return bio_tags_fixed['words_and_tags']

def output_data(all_labels, community, relevant_feats_df):
    curr_cols = relevant_feats_df.columns
    assert len(all_labels) == len(relevant_feats_df)
    relevant_feats_df['yi'] = all_labels
    cols_order = curr_cols.insert(1, 'yi')
    relevant_feats_df = relevant_feats_df[cols_order]
    fpath = output_dir + os.sep + community + '_labels.csv'
    relevant_feats_df.to_csv(fpath, index=False, encoding='utf-8-sig')


def print_fns_stats(all_labeled_terms_without_matches, community, could_be_matched):
    # print(f"Original all_labeled_terms_without_matches len: {len(all_labeled_terms_without_matches)}")
    all_labeled_terms_without_matches = [x for x in all_labeled_terms_without_matches if len(x['term']) > 3]
    global missed_labels_counter
    comm_counter = Counter([t['term'] for t in all_labeled_terms_without_matches if t['term'] not in all_cuiless_terms])
    missed_labels_counter += comm_counter
    # print(f"*** {community} Counter {len(comm_counter)}***")


def iterate_labels_calculate_stats(high_recall_matches_matched_for_file_name, labels_df, relevant_feats_df, community):
    could_be_matched = 0
    all_labeled_terms_without_matches = []
    for label_row_idx, label_row in labels_df.iterrows():
        could_be_matched, unmatched_labels = iterate_row(community, could_be_matched,
                                                         high_recall_matches_matched_for_file_name, label_row,
                                                         relevant_feats_df)

        all_labeled_terms_without_matches += unmatched_labels
    return all_labeled_terms_without_matches, could_be_matched


def iterate_row(community, could_be_matched, high_recall_matches_matched_for_file_name, label_row, relevant_feats_df):
    matches_for_filename = high_recall_matches_matched_for_file_name[label_row['file_name']]
    unmatched_labels = leave_only_unmatched_labels(matches_for_filename, label_row[FINAL_LABELS_COL],
                                                   label_row['file_name'])
    high_recall_matches_lst = get_high_recall_matches_lst_for_post(label_row, relevant_feats_df)
    file_name = label_row['file_name']
    could_be_matched = calculate_could_be_matched(could_be_matched, high_recall_matches_lst, label_row,
                                                  unmatched_labels)
    print_unmatched_labels = False
    if print_unmatched_labels:
        if len(unmatched_labels) > 0:
            print(f"file_name: {label_row['file_name']}")
            print(unmatched_labels)
            print("\n")
    return could_be_matched, unmatched_labels


def calculate_could_be_matched(could_be_matched, high_recall_matches_lst, label_row, unmatched_labels):
    for l in unmatched_labels:
        l_could_be_matched = False
        possible_matches = []
        for m in high_recall_matches_lst:
            if words_similarity(l['term'], m['term']) > LOW_SIMILARITY_THRESHOLD:
                l_could_be_matched = True
                m['type'] = 'high_rec'
                possible_matches.append(m)
                could_be_matched += 1
        if l_could_be_matched:
            print_could_be_matched = False
            if print_could_be_matched:
                print(f'l_could_be_matched, file_name: {label_row["file_name"]}')
                print(l)
                for s in possible_matches:
                    print(s)
                print()
    return could_be_matched


def get_labels_for_high_recall_matches(extract_feats_df, labels_df, relevant_feats_df):
    high_recall_matches_matched_for_file_name = defaultdict(list)
    ### DEBUG ###
    debug = False
    if debug:
        print("DEBUG")
        file_name_to_debug = 127
        relevant_feats_df = relevant_feats_df[relevant_feats_df['file_name'] == file_name_to_debug]
        labels_df = labels_df[labels_df['file_name'] == file_name_to_debug]
    all_labels = []
    for r_idx, row in relevant_feats_df.iterrows():
        label_row = labels_df[labels_df['file_name'] == row['file_name']].iloc[0]

        label_row_tokenized_txt = replace_puncs(label_row['tokenized_text'])
        label_row_tokenized_text_single_spaces = " ".join(
            [x for x in label_row_tokenized_txt.split(" ") if x != ' ' and x != ''])
        row_tokenized_text_single_spaces = " ".join(
            [x for x in replace_puncs(row['tokenized_text']).split(" ") if x != ' ' and x != ''])

        assert label_row_tokenized_text_single_spaces == row_tokenized_text_single_spaces

        if row['match_type'] == 'lemmas':
            yi, best_match = get_yi_for_lemma(extract_feats_df, label_row, r_idx, row)
        else:
            yi, best_match = get_yi_for_cand_match(label_row, row)

        if yi == 1:
            matches_for_post = high_recall_matches_matched_for_file_name[row['file_name']]
            not_intersecting = True
            row_start_offset = row['curr_occurence_offset']
            row_end_offset = row['curr_occurence_offset'] + len(row['cand_match'])
            for m in matches_for_post:
                if m['best_match']['term'] == best_match['term'] and spans_match(m['best_match'], row_start_offset,
                                                                                 row_end_offset):
                    not_intersecting = False
            if not_intersecting:
                high_recall_matches_matched_for_file_name[row['file_name']].append(
                    {'cand_match': row['cand_match'], 'best_match': best_match})

        all_labels.append(yi)
    return all_labels, high_recall_matches_matched_for_file_name, labels_df, relevant_feats_df


def get_high_recall_matches_lst_for_post(label_row, relevant_feats_df):
    high_recall_matches_lst = []
    for _, r in relevant_feats_df[relevant_feats_df['file_name'] == label_row['file_name']][
        ['cand_match', 'curr_occurence_offset', 'all_match_occ']].iterrows():
        d = {'term': r['cand_match'], 'start_offset': r['curr_occurence_offset'],
             'end_offset': r['curr_occurence_offset'] + len(r['cand_match']), 'all_match_occ': r['all_match_occ']}
        high_recall_matches_lst.append(d)
    return high_recall_matches_lst

def leave_only_unmatched_labels(matches_for_filename, post_labeled_terms, file_name):
    for m in matches_for_filename:
        if m['best_match'] not in post_labeled_terms:
            most_similar_t = None
            best_sim = -1
            for t in post_labeled_terms:
                sim = words_similarity(t['term'], m['best_match']['term'])
                if sim >= LOW_SIMILARITY_THRESHOLD and sim > best_sim:
                    most_similar_t = t
                    best_sim = sim
            if most_similar_t:
                post_labeled_terms.remove(most_similar_t)
            else:
                print(f"*** Couldn't remove {m}, file_name: {file_name}")
        else:
            post_labeled_terms.remove(m['best_match'])
    return post_labeled_terms


def term_contains_more_then_3_chars(t):
    t_no_puncs = replace_puncs(deepcopy(t)).replace(" ", "")
    return len(t_no_puncs) > 3

def strip_terms(lst):
    for t in lst:
        t['term'] = t['term'].strip().lower()
    lst = [t for t in lst if term_contains_more_then_3_chars(t['term'])]
    return lst

def spans_match(ann, row_start_offset, row_end_offset):
    return abs(ann['start_offset'] - row_start_offset) <= 3 and abs(ann['end_offset'] - row_end_offset) <= 3

def spans_match_lemma(ann, row_start_offset, row_end_offset):
    return abs(ann['start_offset'] - row_start_offset) <= 4 and abs(ann['end_offset'] - row_end_offset) <= 4


def get_yi_for_cand_match(label_row, row):
    row_start_offset = row['curr_occurence_offset']
    row_end_offset = row['curr_occurence_offset'] + len(row['cand_match'])
    found_label_for_row = False
    best_match = None
    for ann in label_row[FINAL_LABELS_COL]:
        if spans_match(ann, row_start_offset, row_end_offset):
            if words_similarity(ann['term'], row['cand_match']) >= LOW_SIMILARITY_THRESHOLD:
                found_label_for_row = True
                best_match = ann

    yi = 1 if found_label_for_row else 0
    return yi, best_match


def get_yi_for_lemma(extract_feats_df, label_row, r_idx, row):
    best_match, best_match_sim = get_best_match(label_row, row, FINAL_LABELS_COL)
    if best_match:
        extract_feats_df.at[r_idx, 'match'] = best_match['term']
        yi = 1
    else:
        yi = 0
    return yi, best_match


def get_best_match(relevant_labeled_df, row, final_label_col):
    best_match = None
    best_match_sim = 0
    row_start_offset = row['curr_occurence_offset']
    row_end_offset = row_start_offset + len(row['cand_match'])
    for match_data in relevant_labeled_df[final_label_col]:
        term = match_data['term']
        sim1 = words_similarity(term, row['cand_match'])
        sim2 = words_similarity(term, row['umls_match'])
        higher_sim = max(sim1, sim2)
        if higher_sim > LOW_SIMILARITY_THRESHOLD and spans_match_lemma(match_data, row_start_offset, row_end_offset):
            best_match_sim = higher_sim
            best_match = match_data
    return best_match, best_match_sim

if __name__ == '__main__':
    handle_community('sclerosis')
    handle_community('depression')
    handle_community('diabetes')

    print("Adding labels - Done.")
