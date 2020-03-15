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

            for match in row['matches_found']:
                # get the matches indexes in text
                occurences_indexes_in_txt_words = self.get_all_occurences_of_match_in_text(match['umls_match'], txt_words)

                # create windows
                for match_occurence_idx_in_txt_words in occurences_indexes_in_txt_words:
                    match_3_window, match_6_window, match_10_window = self.get_windows_for_match(txt_words, match_occurence_idx_in_txt_words)

                    match_data = {'umls_match': match['umls_match'], 'cand_match': match['cand_match'],
                                  'file_name': row['file_name'], 'row_idx': row_idx,
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