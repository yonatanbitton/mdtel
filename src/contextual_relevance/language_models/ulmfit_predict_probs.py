import difflib
import pandas as pd
import os
import random
from fastai import *
from fastai.text import *

input_dir = r'E:\mdtel_data\data\contextual_relevance'
output_dir = r'E:\mdtel_data\data\contextual_relevance\language_models\output'

def predict_row(row, loaded_learn):
    match = row['match']
    match_10_window = row['match_10_window']
    match_6_window = row['match_6_window']
    match_3_window = row['match_3_window']

    p_10_window = loaded_learn.get_prob_of_word_in_context(match_10_window, match).item()
    p_6_window = loaded_learn.get_prob_of_word_in_context(match_6_window, match).item()
    p_3_window = loaded_learn.get_prob_of_word_in_context(match_3_window, match).item()

    return p_10_window, p_6_window, p_3_window


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

            for match in matches_words:
                # get the matches indexes in text
                match_indexes = self.get_matches(match, txt_words)

                # create windows
                for idx in match_indexes:
                    match_3_window, match_6_window, match_10_window = self.get_windows_for_match(txt_words, idx)

                    match_data = {'match': txt_words[idx], 'match_3_window': match_3_window,
                                  'match_6_window': match_6_window,
                                  'match_10_window': match_10_window}
                    train_instances.append(match_data)
        df = pd.DataFrame(train_instances)
        print(f"WindowsMaker finished with community. Got cols: {df.columns}")
        print(df.head(3))
        return df

def handle_community(community, loaded_learn=None):
    print(f"community: {community}")
    df = pd.read_excel(input_dir + os.sep + community + ".xlsx")

    windows_maker = WindowsMaker()
    df = windows_maker.go(df)
    print("Going to pred")
    preds_10_window = []
    preds_6_window = []
    preds_3_window = []
    for row_idx, row in df.iterrows():
        if row_idx % 10 == 0:
            print(f"row_idx: {row_idx}, df len: {len(df)}")
        p_10_window, p_6_window, p_3_window = predict_row(row, loaded_learn)
        preds_10_window.append(p_10_window)
        preds_6_window.append(p_6_window)
        preds_3_window.append(p_3_window)

    df['pred_10_window'] = preds_10_window
    df['pred_6_window'] = preds_6_window
    df['pred_3_window'] = preds_3_window
    cols = list(df.columns)
    print(f"cols: {cols}, {len(cols)}")
    fpath = output_dir + os.sep + community + '_output.csv'
    print(f"Writing file at shape: {df.shape} to fpath: {fpath}")
    df.to_csv(fpath, index=False)


def get_lm():
    loaded_learn = load_learner(input_dir + os.sep + "language_models", 'ulmfit_lm.pickle')
    TEXT = "במהלך השנה 1948 קמה מדינת ישראל"
    N_WORDS = 40
    N_SENTENCES = 1
    print("\n".join(loaded_learn.predict(TEXT, N_WORDS, temperature=0.9) for _ in range(N_SENTENCES)))
    return loaded_learn


loaded_learn = get_lm()
handle_community('diabetes', loaded_learn)
handle_community('sclerosis', loaded_learn)
handle_community('depression', loaded_learn)

print("Done")