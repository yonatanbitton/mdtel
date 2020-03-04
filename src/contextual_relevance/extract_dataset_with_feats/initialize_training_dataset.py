import difflib
import os
import pandas as pd
import json
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

input_dir = data_dir + r"contextual_relevance\posts"
output_dir = data_dir + r"contextual_relevance\initialized_training_dataset"

if DEBUG:
    print(f"*** DEBUG MODE: Taking 100 rows only ***")

class WindowsMaker:
    def __init__(self):
        print("WindowsMaker Initialized")

    def words_similarity(self, a, b):
        seq = difflib.SequenceMatcher(None, a, b)
        return seq.ratio()

    def get_matches(self, match, txt_words):
        match_indexes = []
        for idx, w in enumerate(txt_words):
            if self.words_similarity(w, match) > 0.85:
                match_indexes.append(idx)
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
            txt = row['tokenized_txt']
            if str(txt) == 'nan':
                continue
            txt_words = txt.split(" ")
            matches_found = row['matches_found']
            matches_words = [match[0].split('-')[0] for match in matches_found]

            for match_idx, match in enumerate(matches_words):
                # get the matches indexes in text
                match_indexes = self.get_matches(match, txt_words)

                # create windows
                for idx in match_indexes:
                    match_3_window, match_6_window, match_10_window = self.get_windows_for_match(txt_words, idx)

                    match_data = {'match': txt_words[idx], 'row_idx': row_idx, 'match_idx': match_idx, 'tokenized_txt': row['tokenized_txt'], 'match_3_window': match_3_window,
                                  'match_6_window': match_6_window,
                                  'match_10_window': match_10_window}
                    train_instances.append(match_data)
        df = pd.DataFrame(train_instances)
        print(f"WindowsMaker finished with community. Got cols: {df.columns}")
        print(df.head(3))
        return df

def handle_community(community):
    print(f"community: {community}")
    df = pd.read_excel(input_dir + os.sep + community + ".xlsx")

    if DEBUG:
        df = df.head(100)

    windows_maker = WindowsMaker()
    df = windows_maker.go(df)

    fpath = output_dir + os.sep + community + '.xlsx'
    print(f"Writing file at shape: {df.shape} to fpath: {fpath}")
    df.to_excel(fpath, index=False)


if __name__ == '__main__':
    handle_community('diabetes')
    handle_community('sclerosis')
    handle_community('depression')

    print("Done")