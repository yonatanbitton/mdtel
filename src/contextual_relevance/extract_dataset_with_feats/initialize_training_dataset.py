import difflib
import os
import pandas as pd
import json
import sys

from debug.high_recall_matcher import word_is_english

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

# input_dir = data_dir + r"contextual_relevance\posts"
input_dir = data_dir + r"high_recall_matcher\output"
output_dir = data_dir + r"contextual_relevance\initialized_training_dataset"

# DEBUG = True
SIMILARITY_THRESHOLD = 0.85

class WindowsMaker:
    def __init__(self):
        print("WindowsMaker Initialized")

    def words_similarity(self, a, b):
        seq = difflib.SequenceMatcher(None, a, b)
        return seq.ratio()

    def match_not_similar_to_short_match(self, match, all_short_matches):
        similar = False
        for m in all_short_matches:
            if self.words_similarity(m, match) > SIMILARITY_THRESHOLD:
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
            if self.words_similarity(cand_term, match) > SIMILARITY_THRESHOLD:
                matches_with_idx = " ".join(txt_words[gram_idx:gram_idx+len_match])
                if not self.words_similarity(matches_with_idx, match) > SIMILARITY_THRESHOLD:
                    print("FUck", matches_with_idx, match)
                assert self.words_similarity(matches_with_idx, match) > SIMILARITY_THRESHOLD
                match_indexes.append(gram_idx)

        return match_indexes

    def get_prefix(self, idx, k, txt_words):
        if idx - k <= 0:
            return txt_words[:idx]
        return txt_words[idx - k:idx]

    def get_windows_for_match(self, txt_words, idx):
        match_3_window = " ".join(self.get_prefix(idx, 3, txt_words))
        match_6_window = " ".join(self.get_prefix(idx, 6, txt_words))
        match_10_window = " ".join(self.get_prefix(idx, 10, txt_words))
        return match_3_window, match_6_window, match_10_window

    def go(self, community_df):
        community_df['matches_found'] = community_df['matches_found'].apply(json.loads)

        train_instances = []

        for row_idx, row in community_df.iterrows():
            if str(row['tokenized_text']) == 'nan':
                continue
            txt, txt_words = self.get_txt_words(row)

            most_specific_matches = self.get_most_specific_matches(row)

            # if most_specific_matches != row['matches_found']:
            #     print(f"Most specific for: {[m['umls_match'] for m in row['matches_found']]} is {[m['umls_match'] for m in most_specific_matches]}")

            for match in most_specific_matches:
                # get the matches indexes in text
                occurences_indexes_in_txt_words = self.get_all_occurences_of_match_in_text(match['umls_match'], txt_words)

                # create windows
                for match_occurence_idx_in_txt_words in occurences_indexes_in_txt_words:
                    match_3_window, match_6_window, match_10_window = self.get_windows_for_match(txt_words, match_occurence_idx_in_txt_words)

                    if 'match_eng' not in match:
                        match['match_eng'] = match['umls_match']

                    match_data = {'umls_match': match['umls_match'], 'cand_match': match['cand_match'],
                                  'file_name': row['file_name'], 'row_idx': row_idx, 'match_eng': match['match_eng'],
                                  'match_occurence_idx_in_txt_words': match_occurence_idx_in_txt_words,
                                  'occurences_indexes_in_txt_words': occurences_indexes_in_txt_words,
                                  'txt_words': txt_words,
                                  'tokenized_text': txt, 'match_3_window': match_3_window,
                                  'match_6_window': match_6_window, 'match_10_window': match_10_window,
                                  'match_tui': match['match_tui'], 'semantic_type': match['semantic_type'],
                                  'all_match_occ': match['all_match_occ'], 'curr_occurence_offset': match['curr_occurence_offset']}
                    train_instances.append(match_data)

        df = pd.DataFrame(train_instances)
        print(f"WindowsMaker finished with community. Got cols: {df.columns}")
        print(df.head(3))
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
            if other_m['curr_occurence_offset'] == 'lemma' or other_m['curr_occurence_offset'] is None:
                continue
            other_len = len(other_m['umls_match'].split(" "))
            if other_len > curr_len:
                other_m_with_same_len_as_curr = " ".join(other_m['umls_match'].split(" ")[:curr_len])
                if len(other_m['umls_match'].split(" ")[curr_len]) >= 3 and self.words_similarity(other_m_with_same_len_as_curr, m['umls_match']) > 0.88:
                    if m['curr_occurence_offset'] == 'lemma' and curr_len == 1 and other_len == 2:
                        all_instances_of_term_are_contained = self.are_all_instances_of_term_are_contained(m, other_m, row)
                        if all_instances_of_term_are_contained:
                            return False
                    elif m['curr_occurence_offset'] != 'lemma':
                        other_span_contains_curr = self.does_other_span_contains_curr(m, other_m)
                        if other_span_contains_curr:
                            return False
        return True

    def are_all_instances_of_term_are_contained(self, m, other_m, row):
        txt_words = row['tokenized_text'].split(" ")
        indexes_with_term = []
        for w_idx, w in enumerate(txt_words):
            if self.words_similarity(w, m['umls_match']) > SIMILARITY_THRESHOLD:
                indexes_with_term.append(w_idx)
        first_container_w, second_container_w = other_m['umls_match'].split(" ")
        all_instances_of_term_are_contained = all(self.words_similarity(txt_words[i + 1], second_container_w) > SIMILARITY_THRESHOLD for i in indexes_with_term)
        if all_instances_of_term_are_contained:
            print(f"all_instances_of_term_are_contained: {m['umls_match']}, {other_m['umls_match']}")
        return all_instances_of_term_are_contained

    def does_other_span_contains_curr(self, m, other_m):
        s1 = m['curr_occurence_offset']
        s2 = other_m['curr_occurence_offset']
        try:
            e1 = s1 + len(m['cand_match'])
        except Exception as ex:
            print("Hey")
        e2 = s2 + len(other_m['cand_match'])
        if abs(s1 - s2) <= 2 or abs(e1 - e2) <= 2:
            print(f"Found long that contains: {m['umls_match']}, {other_m['umls_match']}, {s1, e1}, {s2, e2}")
            # return False
            return False
        return True


def handle_community(community):
    print(f"community: {community}")

    if DEBUG:
        df = pd.read_csv(input_dir + os.sep + community + "_debug.csv")
    else:
        df = pd.read_csv(input_dir + os.sep + community + ".csv")

    windows_maker = WindowsMaker()
    df = windows_maker.go(df)

    fpath = output_dir + os.sep + community + '.csv'
    print(f"Writing file at shape: {df.shape} to fpath: {fpath}")
    df.to_csv(fpath, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    handle_community('sclerosis')
    handle_community('diabetes')
    handle_community('depression')

    print("Done")