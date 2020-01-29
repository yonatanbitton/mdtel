
import difflib
import json
import os
import pickle
from collections import Counter, defaultdict

import pandas as pd
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from contextual_relevance.yap.yap_api import YapApi

SMALL_DATASET = False
DEBUG = False

if SMALL_DATASET:
    detection_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\manual_tag_iter_before_presentation'
else:
    detection_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls'

output_with_features_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\with_features_part_1'
output_with_ulmfit_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\with_ulm_fit_part_2'
output_dir_all_features = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\with_all_features'
output_with_predictions = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\with_predictions'
output_final_matches = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\final_matches'

"""
This file does:
1. prepare_dataset_of_detection_windows
2. Add_count_features
3. Add lm features - Should be in VM - Exceptional case.
4. Add_ngram_lm_features
5. Add relatedness features
6. Predict with classifier
"""

def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


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

            if SMALL_DATASET:
                if str(row['manual_tag']) == 'nan':
                    # print(f'no manual label: {txt}\n')
                    continue
                else:
                    manual_labels = [x.strip() for x in row['manual_tag'].split(",")]

            for match in matches_words:
                # get the matches indexes in text
                match_indexes = self.get_matches(match, txt_words)

                # create windows
                for idx in match_indexes:
                    match_3_window, match_6_window, match_10_window = self.get_windows_for_match(txt_words, idx)

                    if SMALL_DATASET:
                        yi = 0
                        for tag in manual_labels:
                            if words_similarity(tag, match) > 0.85:
                                yi = 1
                                break

                    match_data = {'match': txt_words[idx], 'match_3_window': match_3_window,
                                  'match_6_window': match_6_window,
                                  'match_10_window': match_10_window,
                                  'row_idx': row_idx}

                    if SMALL_DATASET:
                        match_data.update({'yi': yi})

                    train_instances.append(match_data)

                    # if yi == 0:
                    #     print(f'row_idx: {row_idx + 2}, txt: {" ".join(txt_words[:6])} match: {match}, match_6_window: {match_6_window}\n\n')
        df = pd.DataFrame(train_instances)
        print(f"WindowsMaker finished with community. Got cols: {df.columns}")
        print(df.head(3))
        return df

class CountFeatureExtractor:
    def __init__(self, lazy=False):
        if lazy:
            pass
        else:
            self.wiki_data_dir = r'E:\ddrive_desktop\ThesisRelated\Yonatan_Hebrew_ULMFiT\data\wiki_train_dev_data'
            self.all_words_counter = self.prepare_wiki_data_counter()
            self.number_of_unique_tokens = len(self.all_words_counter)
            print(f"CountFeatureExtractor Initialized")

    def prepare_wiki_data_counter(self):
        train_data_path = self.wiki_data_dir + os.sep + 'train' + os.sep + 'train.txt'
        valid_data_path = self.wiki_data_dir + os.sep + 'valid' + os.sep + 'valid.txt'
        with open(train_data_path, encoding='utf-8') as f:
            train_lines = [x.rstrip('\n') for x in f.readlines()]
        with open(valid_data_path, encoding='utf-8') as f:
            valid_lines = [x.rstrip('\n') for x in f.readlines()]
        wiki_lines = train_lines + valid_lines

        if DEBUG:
            wiki_lines = wiki_lines[:100]

        all_words = []
        for line in wiki_lines:
            line_words = line.split(' ')
            all_words += line_words

        all_words_counter = Counter(all_words)

        print(f"got {len(all_words)} words, of which {len(set(all_words))} unique.")
        return all_words_counter

    def go(self, comm_df):
        all_match_counts = []
        all_match_freqs = []
        for row_idx, row in comm_df.iterrows():
            match_count = self.all_words_counter[row['match']]
            match_freq = match_count / self.number_of_unique_tokens
            all_match_counts.append(match_count)
            all_match_freqs.append(match_freq)
            if row_idx % 100 == 0:
                print(row_idx, "out of", len(comm_df), "CountFeatureExtractor", row['match'], match_count, match_freq)
        comm_df['match_count'] = all_match_counts
        comm_df['match_freq'] = all_match_freqs
        print(f"CountFeatureExtractor finished with community. Got cols: {comm_df.columns}")
        print(comm_df.head(3))
        return comm_df

def dd2():
    return 0

def dd():
    return defaultdict(dd2)

class NgramExtractor:
    def __init__(self, lazy=False):
        if lazy:
            pass
        else:
            self.ngram_model = self.get_lm()
            print(f"NgramExtractor Initialized")

    def get_lm(self):
        ngram_model_path = r"E:\ngram_models_HUGE_two_gram.pickle"

        with open(ngram_model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        two_gram_model = loaded_model['two_gram_model']
        print(dict(two_gram_model[('מתמטיים', 'חדשים')]))
        ngram_model = two_gram_model

        print(f"Number of grams: {len(ngram_model.keys())}")
        return ngram_model

    def predict_row(self, row, loaded_learn):
        match = row['match']
        match_10_window = row['match_10_window']
        match_6_window = row['match_6_window']
        match_3_window = row['match_3_window']

        p_10_window = loaded_learn.get_prob_of_word_in_context(match_10_window, match).item()
        p_6_window = loaded_learn.get_prob_of_word_in_context(match_6_window, match).item()
        p_3_window = loaded_learn.get_prob_of_word_in_context(match_3_window, match).item()

        return p_10_window, p_6_window, p_3_window

    def go(self, df):
        preds = []
        for row_idx, row in df.iterrows():
            if str(row['match_3_window']) == 'nan':
                pred = -1
            else:
                context = tuple(row['match_3_window'].split(" ")[1:])
                pred = self.ngram_model[context][row['match']]
                if row_idx % 100 == 0:
                    print(f"{row_idx}, out of {len(df)}, NgramExtractor, context: {context}, pred: {pred}")
            preds.append(pred)

        col_name = "pred_2_gram"

        df[col_name] = preds
        print(f"NgramExtractor finished with community. Got cols: {df.columns}")
        print(df.head(3))
        return df

class RelatednessExtractor:
    def __init__(self, lazy=False):
        if lazy:
            pass
        else:
            self.heb_to_eng_output_path = r"D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\db_data\HEB_TO_ENG_MRCONSO_RELATEDNESS.csv"
            self.similarity_threshold = 0.7

            heb_searcher, umls_data = self.get_umls_data()
            self.heb_searcher = heb_searcher
            self.umls_data = umls_data

            self.yap = YapApi()
            self.ip = '127.0.0.1:8000'
            self.no_relatedness_counter = 0
            print(f"RelatednessExtractor Initialized")

    def get_umls_data(self):
        umls_data = pd.read_csv(self.heb_to_eng_output_path)
        print(f"Got UMLS data at length {len(umls_data)}")
        umls_data = umls_data[~umls_data['STR'].isna()]
        print(f"After filter {len(umls_data)}")

        umls_data.sort_values(by=['HEB'], inplace=True)
        umls_data = umls_data[umls_data['STR'].apply(lambda x: len(x) > 4)]
        umls_data = umls_data[umls_data['HEB'].apply(lambda x: len(x) > 3)]
        umls_data.reset_index(inplace=True)

        heb_umls_list = list(umls_data['HEB'].values)

        heb_db = DictDatabase(CharacterNgramFeatureExtractor(2))

        for heb_w in heb_umls_list:
            heb_db.add(heb_w)
        heb_searcher = Searcher(heb_db, CosineMeasure())

        umls_data.set_index('HEB', inplace=True)

        return heb_searcher, umls_data

    def find_indexes_of_start(self, md_lattice):
        indexes_of_starts = []
        for idx, val in enumerate(md_lattice['num_last'].values):
            if val == 1:
                indexes_of_starts.append(idx)
        return indexes_of_starts

    def build_dataframes(self, indexes_of_starts, md_lattice):
        pairs = list(zip(indexes_of_starts, indexes_of_starts[1:]))
        dataframes = []
        for pair in pairs:
            df = md_lattice.iloc[pair[0]:pair[1]]
            dataframes.append(df)
        last_df = md_lattice.iloc[pairs[-1][1]:]
        dataframes.append(last_df)
        return dataframes

    def get_dataframes_of_words_and_lemmas(self, md_lattice):
        if 'num_last' not in md_lattice:
            return []
        md_lattice['num_last'] = md_lattice['num_last'].apply(lambda x: int(x))
        indexes_of_starts = self.find_indexes_of_start(md_lattice)
        if len(indexes_of_starts) == 1:
            dataframes = [md_lattice]
        else:
            dataframes = self.build_dataframes(indexes_of_starts, md_lattice)
        return dataframes

    def max_len(self, s):
        return max(s, key=len)

    def get_all_words_and_lemmas_from_df(self, dataframes):
        all_words_and_lemmas = []
        for df in dataframes:
            words_and_lemmas_groups = df.groupby('num_last').agg({'lemma': self.max_len, 'word': self.max_len})
            words_and_lemmas_groups = words_and_lemmas_groups.sort_values(by=['num_last'])
            words_and_lemmas = [x['lemma'] + " " + x['word'] for x in
                                words_and_lemmas_groups.to_dict('index').values()]
            all_words_and_lemmas += words_and_lemmas
        return all_words_and_lemmas

    def get_match_word_and_lemma(self, row, match_to_find_relatedness):
        match_with_context = str(row['match_10_window']) + " " + str(match_to_find_relatedness)
        tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = self.yap.run(match_with_context, self.ip)
        dataframes = self.get_dataframes_of_words_and_lemmas(md_lattice)
        words_and_lemmas = self.get_all_words_and_lemmas_from_df(dataframes)
        match_word, match_lemma = words_and_lemmas[-1].split(" ")
        return match_word, match_lemma

    def return_match_or_none(self, heb_searcher, match_cand, similarity_threshold):
        search_result = heb_searcher.ranked_search(match_cand, similarity_threshold)
        if len(search_result) > 0:
            return search_result[0]
        else:
            return None

    def find_source_in_umls(self, match_to_find_relatedness, row):
        match_in_umls = self.return_match_or_none(self.heb_searcher, match_to_find_relatedness, self.similarity_threshold)
        if match_in_umls is None:
            match_word, match_lemma = self.get_match_word_and_lemma(row, match_to_find_relatedness)
            match_in_umls = self.return_match_or_none(self.heb_searcher, match_word, self.similarity_threshold)
            if match_in_umls is None:
                match_in_umls = self.return_match_or_none(self.heb_searcher, match_lemma, self.similarity_threshold)
        return match_in_umls

    def go(self, community_df, community):

        self.term_to_relatedness_df = self.create_term_to_relatedness_df(community)

        new_rows = []
        for row_idx, row in community_df.iterrows():
            if row_idx % 100 == 0:
                print(f"self.no_relatedness_counter: {self.no_relatedness_counter }, idx: {row_idx}, out of: {len(community_df)}")
            match_to_find_relatedness = row['match']
            comm_relatedness = self.get_relatedness_of_match(community, match_to_find_relatedness, row)
            new_row = row.to_dict()
            new_row['relatedness'] = comm_relatedness
            new_rows.append(new_row)
        new_df = pd.DataFrame(new_rows)
        print(f"RelatednessExtractor finished with community. Got cols: {new_df.columns}")
        print(new_df.head(3))
        return new_df

    def parse_line(self, line):
        try:
            relatedness, _, term = line.split("<>")
            index_of_bracket = str(term).index('(')
        except Exception:
            print(f"problem with line: {line}")
        term = term[:index_of_bracket]

        return (term, relatedness)

    def create_term_to_relatedness_df(self, community):
        relatedness_paths = r'E:\umls_perl\UMLS-Similarity-1.47\utils\combined_tty_and_without\after_vietnam\no_dups'
        with open(relatedness_paths + os.sep + community + "_output_no_dups.txt", encoding='utf-8') as f:
            lines = f.readlines()

        all_terms = []
        all_relatedness = []
        for line in lines:
            term, relatedness = self.parse_line(line)
            all_terms.append(term.lower())
            all_relatedness.append(relatedness)
        term_and_relatedness_df = pd.DataFrame()
        term_and_relatedness_df['term'] = all_terms
        term_and_relatedness_df['relatedness'] = all_relatedness

        term_and_relatedness_df['term_prefix'] = term_and_relatedness_df['term'].apply(lambda x: x[:3])

        return term_and_relatedness_df

    def is_english(self, s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def get_relatedness_of_close_line(self, term, relatedness, match):
        if words_similarity(term.lower(), match.lower())>0.9:
            return relatedness

    def get_relatedness_of_match(self, community, match_to_find_relatedness, row):
        match_in_umls = self.find_source_in_umls(match_to_find_relatedness, row)
        if match_in_umls is None:
            if self.is_english(row['match']):
                comm_relatedness = self.try_to_find_english_relatedness(row)
                if comm_relatedness:
                    return comm_relatedness
            print(f"match: {match_to_find_relatedness}, relatedness wasn't found!")
            self.no_relatedness_counter += 1
            comm_relatedness = -1
        else:
            match_word = match_in_umls[1]
            subset_df_of_match = self.umls_data.loc[match_word]
            if type(self.umls_data.loc[match_word]) == pd.DataFrame:
                subset_df_of_match = subset_df_of_match.iloc[0]
            comm_relatedness = subset_df_of_match[community + "_relatedness"]
        return comm_relatedness

    def try_to_find_english_relatedness(self, row):
        found_relatedness = None
        match = row['match'].lower()
        match_prefix = match[:3]
        subset_mask = self.term_to_relatedness_df['term_prefix'].apply(lambda x: x == match_prefix)
        term_to_relatedness_df_subset = self.term_to_relatedness_df[subset_mask]
        for line_idx, cand_line in term_to_relatedness_df_subset.iterrows():
            term, relatedness = cand_line['term'], cand_line['relatedness']
            if words_similarity(term.lower(), row['match'].lower()) > 0.9:
                found_relatedness = relatedness
        return found_relatedness

    def get_relatedness_of_match2(self, community, match_to_find_relatedness, row):
        match_in_umls = self.find_source_in_umls(match_to_find_relatedness, row)
        if match_in_umls is None:
            print(f"match: {match_to_find_relatedness}, relatedness wasn't found!")
            comm_relatedness = -1
        else:
            match_word = match_in_umls[1]
            subset_df_of_match = self.umls_data.loc[match_word]
            if type(self.umls_data.loc[match_word]) == pd.DataFrame:
                subset_df_of_match = subset_df_of_match.iloc[0]
            comm_relatedness = subset_df_of_match[community + "_relatedness"]
        return comm_relatedness



class Predictor:
    def __init__(self):
        self.output_models_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\trained_classification_models'

    def go(self, community):
        # comm_df = pd.read_excel(output_dir_all_features + os.sep + community + "_all_feats.xlsx")
        comm_df = pd.read_csv(output_dir_all_features + os.sep + community + "_all_feats.csv")

        with open(self.output_models_dir + os.sep + "trained_models.pickle", 'rb') as f:
            models_data_input = pickle.load(f)

        comm_data = models_data_input[community]
        X, df, hard_y_pred = self.predict(comm_data, comm_df)

        comm_df['hard_y_pred'] = hard_y_pred

        assert len(comm_df) == len(hard_y_pred)

        # output_comm_path = output_with_predictions + os.sep + community + "_with_preds.xlsx"
        output_comm_path = output_with_predictions + os.sep + community + "_with_preds.csv"
        print(f"Writing to {output_comm_path}")
        if 'match.1' in comm_df:
            comm_df.drop(columns=['match.1'], inplace=True)

        # comm_df.to_excel(output_comm_path, index=False)
        comm_df.to_csv(output_comm_path, index=False)

        return hard_y_pred

    def predict(self, models_data_input, df):
        # subset = models_data_input['subset']
        all_feats = list((
            'match_count', 'match_freq', 'pred_2_gram', 'pred_3_window', 'pred_6_window', 'pred_10_window',
            'relatedness'))
        subset = all_feats
        threshold = models_data_input['threshold']
        df = df[subset]
        X = df[subset]
        X.fillna(-1, inplace=True)
        model = models_data_input['model']
        y_pred = [x[1] for x in model.predict_proba(X)]
        hard_y_pred = [x > threshold for x in y_pred]
        return X, df, hard_y_pred


def handle_community(community, windows_maker, count_fe, ngram_ext, relatedness_ext):
    print(f"Handling community {community}")
    path = detection_dir + os.sep + community + '.xlsx'
    if SMALL_DATASET:
        path = detection_dir + os.sep + community + '_debug.xlsx'

    community_df = pd.read_excel(path)
    if DEBUG:
        community_df = community_df.sample(1000)

    if windows_maker:
        community_df = windows_maker.go(community_df)
        print("windows_maker done")
        del windows_maker

    if count_fe:
        community_df = count_fe.go(community_df)
        print("count_fe done")
        del count_fe

    if ngram_ext:
        community_df = ngram_ext.go(community_df)
        print("ngram_ext done")
        del ngram_ext

    if relatedness_ext:
        community_df = relatedness_ext.go(community_df, community)
        print("relatedness_ext done")
        del relatedness_ext

    print(f"Finished with community {community}, df len: {len(community_df)}")
    if SMALL_DATASET:
        df_path = output_with_features_dir + os.sep + community + "_features_debug.xlsx"
    else:
        df_path = output_with_features_dir + os.sep + community + "_features.xlsx"
    community_df.to_excel(df_path, encoding='utf-8', index=False)
    print(f"Wrote to path {df_path}")


def part_1_all_features_except_lm():
    windows_maker = WindowsMaker()
    count_fe = CountFeatureExtractor()
    ngram_ext = NgramExtractor()
    relatedness_ext = RelatednessExtractor()
    handle_community('diabetes', windows_maker, count_fe, ngram_ext, relatedness_ext)
    handle_community('sclerosis', windows_maker, count_fe, ngram_ext, relatedness_ext)
    handle_community('depression', windows_maker, count_fe, ngram_ext, relatedness_ext)
    print(f'Finished part 1')



class ULMFitExtractor:
    def go(self, community):
        print(f"ULMFitExtractor with {community}")
        rows_not_found = 0
        if SMALL_DATASET:
            community_part_1_path = output_with_features_dir + os.sep + community + "_features_debug.xlsx"
        else:
            community_part_1_path = output_with_features_dir + os.sep + community + "_features.xlsx"
        community_part_1_df = pd.read_excel(community_part_1_path)

        community_part_2_path = output_with_ulmfit_dir + os.sep + community + "_output.csv"
        community_part_2_df = pd.read_csv(community_part_2_path)

        new_df = pd.merge(community_part_1_df, community_part_2_df, how='outer',
                          on=['match', 'match_3_window', 'match_6_window', 'match_10_window']).drop_duplicates()

        if SMALL_DATASET:
            path = output_dir_all_features + os.sep + community + "_all_feats_debug.xlsx"
            path_csv = output_dir_all_features + os.sep + community + "_all_feats_debug.csv"
        else:
            path = output_dir_all_features + os.sep + community + "_all_feats.xlsx"
            path_csv = output_dir_all_features + os.sep + community + "_all_feats.csv"
        new_df.to_excel(path, index=False)
        new_df.to_csv(path_csv, index=False)
        print(f"Finished ULMFitExtractor with community {community}")
        return new_df

def part_2_ulmfit_feats():
    print(f"Starting part2")
    ulm_ext = ULMFitExtractor()
    ulm_ext.go("diabetes")
    ulm_ext.go("sclerosis")
    ulm_ext.go("depression")
    print("Done part 2")

class WindowsInverser:
    def __init__(self):
        print("WindowsInverser")
        self.prediction_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\with_predictions'

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

    def go(self, community_df, community):
        print(f"WindowsInverser - Going on {community}")
        community_df['matches_found'] = community_df['matches_found'].apply(json.loads)

        # predictions_df = pd.read_excel(self.prediction_dir + os.sep + community + "_with_preds.xlsx")
        predictions_df = pd.read_csv(self.prediction_dir + os.sep + community + "_with_preds.csv")

        print(f"WindowsInverser: {community}")

        all_true_matches = []
        for row_idx, row in community_df.iterrows():
            row_all_matches = row['matches_found']
            row_true_matches = list(predictions_df[predictions_df['row_idx'] == row_idx]['match'].values)
            row_true_matches_to_add = [x for x in row_all_matches if len(x[0].split("-")[0].split(" ")) > 1 or word_needs_to_stay_word(x, row_true_matches)]
            diff = [x for x in row_all_matches if not len(x[0].split("-")[0].split(" ")) > 1 and not word_needs_to_stay_word(x, row_true_matches)]
            print(f"diff: {row_idx}/{len(community_df)}: {diff}")
            all_true_matches.append(row_true_matches_to_add)

        print("Yes")
        community_df['true_matches'] = all_true_matches

        community_df['matches_found'] = community_df['matches_found'].apply(json.dumps)
        community_df['true_matches'] = community_df['true_matches'].apply(json.dumps)

        community_df.to_csv(output_final_matches + os.sep + community + ".csv", encoding='utf-8', index=False)
        community_df.to_excel(output_final_matches + os.sep + community + ".xlsx", encoding='utf-8', index=False)

        print(f"Finished {output_final_matches + os.sep + community}")
        # print(df.head(3))
        # return df

def word_needs_to_stay_word(x, row_true_matches):
    match_itself = x[0].split("-")[0]
    for item in row_true_matches:
        if words_similarity(item, match_itself) > 0.85:
            return True
    return False

def part_4_reverse_df():
    window_inverse = WindowsInverser()

    # community = "diabetes"
    restore_community("diabetes", window_inverse)
    restore_community("sclerosis", window_inverse)
    restore_community("depression", window_inverse)



def restore_community(community, window_inverse):
    community_df = pd.read_excel(detection_dir + os.sep + community + '.xlsx')

    if DEBUG:
        community_df = community_df.head(1000)

    window_inverse.go(community_df, community)

    print(f"Done with community {community}")


def part_3_predict():
    predictor = Predictor()
    predictor.go("diabetes")
    predictor.go("sclerosis")
    predictor.go("depression")


if __name__ == '__main__':
    windows_maker = count_fe = ngram_ext = relatedness_ext = predictor = None
    part_1_all_features_except_lm()
    part_2_ulmfit_feats()
    part_3_predict()
    part_4_reverse_df()

    print("Done")
