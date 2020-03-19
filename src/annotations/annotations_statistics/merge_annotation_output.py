import json
import os
from copy import deepcopy

import pandas as pd

from config import data_dir
from high_recall_matcher_posts_level import words_similarity

labels_dir = data_dir + r'manual_labeled_v2\doccano'
labels_output = labels_dir + os.sep + "merged_output"
measuring_kappa_dir = labels_dir + os.sep + "csv_files" + os.sep + "measuring_kappa"
posts_dir = data_dir + r"high_recall_matcher\posts\lemlda"

SIMILARITY_THRESHOLD = 0.85

def main():
    sclerosis_entities_stats, sclerosis_lines_stats = merge_comm(community='sclerosis')
    diabetes_entities_stats, diabetes_lines_stats = merge_comm(community='diabetes')
    depression_entities_stats, depression_lines_stats = merge_comm(community='depression')

    lines_stats = pd.DataFrame([sclerosis_lines_stats, diabetes_lines_stats, depression_lines_stats], index=['Sclerosis', 'Diabetes', 'Depression'])
    lines_stats.to_excel(labels_output + os.sep + "annotations_lines_stats.xlsx")

    entities_stats = pd.DataFrame([sclerosis_entities_stats, diabetes_entities_stats, depression_entities_stats], index=['Sclerosis', 'Diabetes', 'Depression'])
    entities_stats.to_excel(labels_output + os.sep + "annotations_entities_stats.xlsx")
    print(entities_stats)
    print("Done")


def terms_are_matched(t1, t2):
    t1 = t1.strip()
    t2 = t2.strip()
    if t1 == t2 or words_similarity(t1, t2) > SIMILARITY_THRESHOLD:
        return True
    return False

def spans_match(t1, t2, s1, s2):
    span_limit = 2
    if abs(s1 - s2) <= span_limit:
        return True
    if t1[0] == ' ':
        s1 += 1
    if t2[0] == ' ':
        s2 += 1
    return abs(s1 - s2) <= span_limit

def merge_lsts(user_5_with_6_not_overlap, user_6_with_5_not_overlap):
    merged_out = user_6_with_5_not_overlap
    for m in user_5_with_6_not_overlap:
        if m not in merged_out:
            merged_out.append(m)
    return merged_out


def get_number_of_overlaps(d):
    if d == {}:
        return 0
    return int((len(d['user_5_with_6_overlap']) + len(d['user_6_with_5_overlap'])) / 2)


def get_inner_join_matched_terms(row):
    user_5_labels = row['user_5_labels']
    user_6_labels = row['user_6_labels']

    matched_terms = []
    for m1 in user_5_labels:
        found_in_user_6 = term_exists_in_lst(m1, user_6_labels)
        if found_in_user_6:
            matched_terms.append(m1)
    return matched_terms


def term_exists_in_lst(m1, user_6_labels):
    found_in_user_6 = False
    for m2 in user_6_labels:
        if terms_are_matched(m1['term'], m2['term']) \
                and spans_match(m1['term'], m2['term'], m1['start_offset'], m2['start_offset']) \
                and spans_match(m1['term'], m2['term'], m1['end_offset'], m2['end_offset']):
            found_in_user_6 = True
            break
    return found_in_user_6


def term_is_overlapping_with_lst(m, lst):
    overlappign_with_lst = False
    for m2 in lst:
        if spans_match(m['term'], m2['term'], m['start_offset'], m2['start_offset']) \
                or spans_match(m['term'], m2['term'], m['end_offset'], m2['end_offset']):
            overlappign_with_lst = True
            break
    return overlappign_with_lst


def get_left_join_terms(first_user_labels, second_user_labels):
    left_join_terms = []
    for m1 in first_user_labels:
        term_is_overlapping = term_is_overlapping_with_lst(m1, second_user_labels)
        if not term_is_overlapping:
            left_join_terms.append(m1)
    return left_join_terms


def outer_join_rows(user_5_left_join_user_6, user_6_left_join_user_5):
    merged = user_5_left_join_user_6 + user_6_left_join_user_5
    assert len(merged) == len(user_5_left_join_user_6) + len(user_6_left_join_user_5)
    return merged


def get_overlaps(row):
    user_5_labels = row['user_5_labels']
    user_6_labels = row['user_6_labels']
    inner_joins = row['inner_join_same_matches']
    outer_joins = row['outer_join_users']

    user_5_overlapped_annotations = get_overlaps_lst(user_5_labels, inner_joins, outer_joins)
    user_6_overlapped_annotations = get_overlaps_lst(user_6_labels, inner_joins, outer_joins)

    overlapped_term_d = {'user_5_with_6_overlap': user_5_overlapped_annotations,
                         'user_6_with_5_overlap': user_6_overlapped_annotations}
    return overlapped_term_d


def get_overlaps_lst(annotation_lst, inner_joins, outer_joins):
    overlapped_annotations = []
    for m in annotation_lst:
        if not term_exists_in_lst(m, inner_joins) and not term_exists_in_lst(m, outer_joins):
        # if m not in inner_joins and m not in outer_joins:
            overlapped_annotations.append(m)
    return overlapped_annotations


def get_overlaps_for_user(x, user_number):
    if user_number == 5:
        return x['user_5_with_6_overlap']
    elif user_number == 6:
        return x['user_6_with_5_overlap']
    else:
        raise Exception("Unknown user number")


def merge_comm(community):
    print(f"Merging community: {community}")
    with open(labels_dir + os.sep + community + "_export.json", encoding='utf-8') as f:
        comm_lines = [json.loads(x) for x in f.readlines()]

    posts_df = pd.read_excel(posts_dir + os.sep + community + "_posts.xlsx")
    annotation_df, comm_lines_stats_ser = build_labels_df(comm_lines, community, posts_df)

    annotation_df['inner_join_same_matches'] = annotation_df.apply(lambda row: get_inner_join_matched_terms(row), axis=1)
    annotation_df['user_5_left_join_user_6'] = annotation_df.apply(lambda row: get_left_join_terms(row['user_5_labels'], row['user_6_labels']), axis=1)
    annotation_df['user_6_left_join_user_5'] = annotation_df.apply(lambda row: get_left_join_terms(row['user_6_labels'], row['user_5_labels']), axis=1)
    annotation_df['outer_join_users'] = annotation_df.apply(lambda row: outer_join_rows(row['user_5_left_join_user_6'], row['user_6_left_join_user_5']), axis=1)
    annotation_df['overlaps'] = annotation_df.apply(lambda row: get_overlaps(row), axis=1)
    annotation_df['merged_inner_and_outer'] = annotation_df.apply(lambda row: outer_join_rows(row['inner_join_same_matches'], row['outer_join_users']), axis=1)

    annotation_df['number_of_user_5_annotations'] = annotation_df['user_5_labels'].apply(lambda lst: len(lst))
    annotation_df['number_of_user_6_annotations'] = annotation_df['user_6_labels'].apply(lambda lst: len(lst))
    annotation_df['number_of_inner_join_same_matches'] = annotation_df['inner_join_same_matches'].apply(lambda lst: len(lst))
    annotation_df['number_user_5_left_join_user_6'] = annotation_df['user_5_left_join_user_6'].apply(lambda lst: len(lst))
    annotation_df['number_user_6_left_join_user_5'] = annotation_df['user_6_left_join_user_5'].apply(lambda lst: len(lst))
    annotation_df['number_outer_join_users'] = annotation_df['outer_join_users'].apply(lambda lst: len(lst))
    annotation_df['number_overlaps'] = annotation_df['overlaps'].apply(lambda d: get_number_of_overlaps(d))
    annotation_df['number_merged_inner_and_outer'] = annotation_df['merged_inner_and_outer'].apply(lambda lst: len(lst))
    places_with_overlaps = annotation_df[annotation_df['number_overlaps'] > 0]

    total_user_5 = annotation_df['number_of_user_5_annotations'].sum()
    total_user_6 = annotation_df['number_of_user_6_annotations'].sum()
    total_inner_join = annotation_df['number_of_inner_join_same_matches'].sum()
    total_5_left_join_6 = annotation_df['number_user_5_left_join_user_6'].sum()
    total_6_left_join_5 = annotation_df['number_user_6_left_join_user_5'].sum()
    total_outer_joins = annotation_df['number_outer_join_users'].sum()
    total_overlaps = annotation_df['number_overlaps'].sum()
    total_merged_inner_and_outer = annotation_df['number_merged_inner_and_outer'].sum()
    total_number_of_rows_with_overlaps = len(places_with_overlaps)

    merge_stats = {'total_user_5': total_user_5, 'total_user_6': total_user_6, 'total_inner_join': total_inner_join,
         'total_5_left_join_6': total_5_left_join_6, 'total_6_left_join_5': total_6_left_join_5,
         'total_outer_joins': total_outer_joins, 'total_overlaps': total_overlaps,
         'total_merged_inner_and_outer': total_merged_inner_and_outer,
                   'total_number_of_rows_with_overlaps': total_number_of_rows_with_overlaps}
    merge_stats_ser = pd.Series(merge_stats)
    print(merge_stats_ser)

    for row_idx, row in places_with_overlaps.iterrows():
        user_5_labels = row['user_5_labels']
        user_6_labels = row['user_6_labels']
        overlaps = row['overlaps']
        merged_inner_and_outer = row['merged_inner_and_outer']
        print('\nuser5')
        # print(user_5_labels)
        for t in user_5_labels:
            print(t)
        print('\nuser6')
        for t in user_6_labels:
            print(t)
        # print(user_6_labels)
        print('\noverlaps')
        # print(overlaps)
        for t,v in overlaps.items():
            print(t, v)
        print('\nmerged')
        # print(merged_inner_and_outer)
        for t in merged_inner_and_outer:
            print(t)
        print("\n\n")

    wanted_cols = ['text', 'tokenized_text', 'file_name', 'merged_inner_and_outer']
    annotation_df = annotation_df[wanted_cols]
    annotation_df['merged_inner_and_outer'] = annotation_df['merged_inner_and_outer'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    annotation_df.to_csv(labels_output + os.sep + community + "_labels.csv", index=False, encoding='utf-8-sig')

    places_with_overlaps['yoav_overlaps_with_ora'] = places_with_overlaps['overlaps'].apply(lambda x: get_overlaps_for_user(x, 5))
    places_with_overlaps['ora_overlaps_with_yoav'] = places_with_overlaps['overlaps'].apply(lambda x: get_overlaps_for_user(x, 6))
    places_with_overlaps = places_with_overlaps[['text', 'yoav_overlaps_with_ora', 'ora_overlaps_with_yoav']]
    for c in ['yoav_overlaps_with_ora', 'ora_overlaps_with_yoav']:
        places_with_overlaps[c] = places_with_overlaps[c].apply(lambda x: json.dumps(x, ensure_ascii=False))

    places_with_overlaps.to_csv(labels_output + os.sep + community + "_overlaps.csv", index=False, encoding='utf-8-sig')

    annotation_df.to_csv(labels_output + os.sep + community + "_labels.csv", index=False, encoding='utf-8-sig')

    print(f"Finished with community: {community}\n\n")
    return merge_stats_ser, comm_lines_stats_ser


def row_annotated_by_at_least_one_user(row):
    return len(row['user_5_labels']) > 0 or len(row['user_6_labels']) > 0

def row_annotated_by_both_users(row):
    return len(row['user_5_labels']) > 0 and len(row['user_6_labels']) > 0

def user_X_but_not_Y(row, X, Y):
    return len(row[f'user_{X}_labels']) > 0 and len(row[f'user_{Y}_labels']) == 0

def build_labels_df(comm_lines, community, posts_df):
    all_line_texts = []
    all_user_5_labels = []
    all_user_6_labels = []
    all_relevant_filenames = []
    all_line_tokenized_texts = []

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
    labels_df['user_5_labels'] = all_user_5_labels
    labels_df['user_6_labels'] = all_user_6_labels
    labels_df['file_name'] = all_relevant_filenames
    labels_df['tokenized_text'] = all_line_tokenized_texts

    labels_df = labels_df[labels_df.apply(lambda row: row_annotated_by_at_least_one_user(row), axis=1)]

    labels_df_copy = deepcopy(labels_df)
    for c in ['user_5_labels', 'user_6_labels']:
        labels_df_copy[c] = labels_df_copy[c].apply(lambda x: json.dumps(x, ensure_ascii=False))
    labels_df_copy.to_csv(measuring_kappa_dir + os.sep + community + "_both_annotators.csv", index=False, encoding='utf-8-sig')

    annotated_by_both_users = labels_df[labels_df.apply(lambda row: row_annotated_by_both_users(row), axis=1)]
    user_5_but_not_6 = labels_df[labels_df.apply(lambda row: user_X_but_not_Y(row, 5, 6), axis=1)]
    user_6_but_not_5 = labels_df[labels_df.apply(lambda row: user_X_but_not_Y(row, 6, 5), axis=1)]

    comm_d = {'lines_annotated_by_at_least_one_user': len(labels_df),
         'lines_annotated_by_both_users': len(annotated_by_both_users),
         'lines_annotated_by_user_5_but_not_6': len(user_5_but_not_6),
         'lines_annotated_by_user_6_but_not_5': len(user_6_but_not_5)}
    comm_lines_stats_ser = pd.Series(comm_d, name=community)
    print(comm_lines_stats_ser)
    print("\n")

    for v in user_6_but_not_5['text'].values:
        is_in_at_least_one_users = v in labels_df['text'].values
        is_in_user_5 = v in user_5_but_not_6['text'].values
        if not is_in_at_least_one_users or is_in_user_5:
            raise Exception("Statistics error")

    for v in user_5_but_not_6['text'].values:
        is_in_at_least_one_users = v in labels_df['text'].values
        is_in_user_6 = v in user_6_but_not_5['text'].values
        if not is_in_at_least_one_users or is_in_user_6:
            raise Exception("Statistics error")

    return labels_df, comm_lines_stats_ser


if __name__ == '__main__':
    main()