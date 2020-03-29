import difflib
import os
import re

import pandas as pd
import json
import sys

# from debug.old_code.high_recall_matcher import word_is_english
from high_recall_matcher_posts_level import word_is_english, replace_puncs
from utils import words_similarity

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

# input_dir = data_dir + r"contextual_relevance\posts"
input_dir = data_dir + r"high_recall_matcher\output"
output_dir = data_dir + r"contextual_relevance\initialized_training_dataset"

# DEBUG = True
SIMILARITY_THRESHOLD = 0.85

class WindowsMaker:
    def match_not_similar_to_short_match(self, match, all_short_matches):
        similar = False
        for m in all_short_matches:
            if words_similarity(m, match) > SIMILARITY_THRESHOLD:
                similar = True
                break
        return not similar

    def get_all_occurences_of_match_in_text(self, match, txt_words):

        match_indexes = []
        NUMBER_OF_GRAMS = 3
        len_match = len(match.split(" "))
        ngrams = list(zip(*[txt_words[i:] for i in range(NUMBER_OF_GRAMS)]))
        if len(ngrams) > 0:
            last_gram = ngrams[-1]
            extra_gram = last_gram[1], last_gram[2], 'PAD'
            ngrams.append(extra_gram)
            extra_gram_2 = last_gram[2], 'PAD', 'PAD'
            ngrams.append(extra_gram_2)

        for gram_idx, gram in enumerate(ngrams):
            cand_term = " ".join(gram[:len_match])
            if words_similarity(cand_term, match) > SIMILARITY_THRESHOLD:
                matches_with_idx = " ".join(txt_words[gram_idx:gram_idx+len_match])
                assert words_similarity(matches_with_idx, match) > SIMILARITY_THRESHOLD
                match_indexes.append(gram_idx)

        return match_indexes

    def get_prefix(self, idx, k, txt_words):
        if idx - k <= 0:
            return txt_words[:idx]
        return txt_words[idx - k:idx]

    def get_windows_for_match(self, match, row):
        post_txt = row['post_txt']
        post_txt = replace_puncs(post_txt)
        curr_occurence_offset = match['curr_occurence_offset']
        # match in text - post_txt[curr_occurence_offset:curr_occurence_offset + len(match['cand_match'])]
        relevant_post_prefix = post_txt[:curr_occurence_offset-1]

        relevant_post_idx_words = [w for w in relevant_post_prefix.split(" ") if w != ' ' and w != '']
        relevant_post_idx_single_spaces = " ".join(relevant_post_idx_words) + " "
        indices_of_spaces = [m.start() for m in re.finditer(' ', relevant_post_idx_single_spaces)]
        if len(indices_of_spaces) > 0:
            pairs = list(zip(indices_of_spaces, indices_of_spaces[1:]))
            pairs.insert(0, (-1, indices_of_spaces[0]))
            all_preceding_ws = [relevant_post_idx_single_spaces[i1+1:i2] for i1, i2 in pairs]
        else:
            all_preceding_ws = relevant_post_idx_words

        match_3_window = " ".join(all_preceding_ws[-3:])
        match_6_window = " ".join(all_preceding_ws[-6:])
        match_10_window = " ".join(all_preceding_ws[-10:])

        # print(post_txt[:curr_occurence_offset + len(match['cand_match']) + 1])
        # print(match_3_window + ' - ' + match['cand_match'])
        # print(match_6_window + ' - ' + match['cand_match'])
        # print(match_10_window + ' - ' + match['cand_match'])
        # print("\n")

        return match_3_window, match_6_window, match_10_window

    def go(self, community_df):
        community_df['matches_found'] = community_df['matches_found'].apply(json.loads)

        train_instances = []
        bad_count = 0

        for row_idx, row in community_df.iterrows():
            if str(row['tokenized_text']) == 'nan':
                continue

            # if row['file_name'] != 425:
            #     continue

            txt, txt_words = self.get_txt_words(row)

            most_specific_matches = self.get_most_specific_matches(row)

            for match in most_specific_matches:
                # get the matches indexes in text
                if match['curr_occurence_offset'] == 'lemma' or match['curr_occurence_offset'] is None or not type(match['curr_occurence_offset']) == int:
                    bad_count += 1
                    print(match)
                    print(f"*** BAD COUNT {match['curr_occurence_offset'], match['cand_match']}, bad_count: {bad_count}")
                    print(row['post_txt'])
                    continue

                match_3_window, match_6_window, match_10_window = self.get_windows_for_match(match, row)

                if 'match_eng' not in match:
                    match['match_eng'] = match['umls_match']

                if word_is_english(match['cand_match']):
                    match_type = 'tokenized_text'
                else:
                    match_type = match['hebrew_key']

                match_data = {'umls_match': match['umls_match'], 'cand_match': match['cand_match'],
                              'file_name': row['file_name'], 'row_idx': row_idx, 'match_eng': match['match_eng'],
                              'txt_words': txt_words, 'match_type': match_type,
                              'tokenized_text': txt, 'match_3_window': match_3_window,
                              'match_6_window': match_6_window, 'match_10_window': match_10_window,
                              'match_tui': match['match_tui'], 'semantic_type': match['semantic_type'],
                              'all_match_occ': match['all_match_occ'], 'curr_occurence_offset': match['curr_occurence_offset'],
                              'post_txt': row['post_txt']}
                train_instances.append(match_data)

        df = pd.DataFrame(train_instances)
        # print(f"WindowsMaker finished with community. Got cols: {df.columns}")
        # print(df.head(3))
        for c in ['all_match_occ', 'txt_words']:
            df[c] = df[c].apply(lambda x: json.dumps(x, ensure_ascii=False))
        return df

    def get_txt_words(self, row):
        txt = row['tokenized_text']
        txt = txt.replace(".", " ")
        txt_words = txt.split(" ")
        txt_words = [x.lower() if word_is_english(x) else x for x in txt_words]
        return txt, txt_words

    def get_most_specific_matches(self, row):
        matches = row['matches_found']
        specific_matches = []
        for m in matches:
            if self.no_longer_match_that_contains_m_in_same_span(matches, m, row):
                specific_matches.append(m)
        return specific_matches

    def no_longer_match_that_contains_m_in_same_span(self, matches, m, row):
        curr_len = len(m['umls_match'].split(" "))
        for other_m in matches:
            if other_m['curr_occurence_offset'] is None:
                print(f"other_m['curr_occurence_offset'] is None")
                continue
            other_len = len(other_m['umls_match'].split(" "))
            if other_len > curr_len:
                other_m_with_same_len_as_curr = " ".join(other_m['umls_match'].split(" ")[:curr_len])
                if len(other_m['umls_match'].split(" ")[curr_len]) >= 3 and words_similarity(other_m_with_same_len_as_curr, m['umls_match']) > 0.88:
                    other_span_contains_curr = self.does_other_span_contains_curr(m, other_m, row['file_name'])
                    if other_span_contains_curr:
                        return False
                other_m_with_same_len_as_curr = " ".join(other_m['umls_match'].split(" ")[curr_len:])
                if len(other_m['umls_match'].split(" ")[curr_len]) >= 3 and words_similarity(other_m_with_same_len_as_curr, m['umls_match']) > 0.88:
                    other_span_contains_curr = self.does_other_span_contains_curr(m, other_m, row['file_name'])
                    if other_span_contains_curr:
                        return False

        return True

    def are_all_instances_of_term_are_contained(self, m, other_m, row):
        txt_words = row['tokenized_text'].split(" ")
        indexes_with_term = []
        for w_idx, w in enumerate(txt_words):
            if words_similarity(w, m['umls_match']) > SIMILARITY_THRESHOLD:
                indexes_with_term.append(w_idx)
        first_container_w, second_container_w = other_m['umls_match'].split(" ")
        all_instances_of_term_are_contained = all(words_similarity(txt_words[i + 1], second_container_w) > SIMILARITY_THRESHOLD for i in indexes_with_term)
        # if all_instances_of_term_are_contained:
        #     print(f"all_instances_of_term_are_contained: {m['umls_match']}, {other_m['umls_match']}")
        return all_instances_of_term_are_contained

    def does_other_span_contains_curr(self, m, other_m, fname):
        s1 = m['curr_occurence_offset']
        s2 = other_m['curr_occurence_offset']
        e1 = s1 + len(m['cand_match'])
        e2 = s2 + len(other_m['cand_match'])
        if abs(s1 - s2) <= 2 or abs(e1 - e2) <= 2:
            print(f"Found long that contains at {fname}: {m['umls_match']}, {other_m['umls_match']}, {s1, e1}, {s2, e2}")
            return True
        return False


def handle_community(community):
    # print(f"community: {community}")

    if DEBUG:
        df = pd.read_csv(input_dir + os.sep + community + "_debug.csv")
    else:
        df = pd.read_csv(input_dir + os.sep + community + ".csv")

    df['stripped_txt'] = df['post_txt'].apply(lambda x: x.strip())
    len_before_drop = len(df)
    df = df.drop_duplicates(subset=['stripped_txt'])

    print(f"Len before drop: {len_before_drop}, after: {len(df)}")

    windows_maker = WindowsMaker()
    df = windows_maker.go(df)

    fpath = output_dir + os.sep + community + '.csv'
    # print(f"Writing file at shape: {df.shape} to fpath: {fpath}")
    df.to_csv(fpath, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    print("Initializing training dataset ...")
    handle_community('depression')
    handle_community('sclerosis')
    handle_community('diabetes')

    print("Training dataset initialized.\n\n")