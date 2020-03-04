import difflib
import os
import sys
import traceback

import pandas as pd
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from contextual_relevance.extract_dataset_with_feats.yap.yap_api import YapApi

module_path = os.path.abspath(os.path.join('..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

input_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
calculated_relatedness_dir = data_dir + r"contextual_relevance\relatedness"
umls_data_path = data_dir + "high_recall_matcher\HEB_TO_ENG_MRCONSO_RELATEDNESS.csv"

output_dir = data_dir + r"contextual_relevance\relatedness\output"
similarity_threshold = 0.70

yap = YapApi()
ip = '127.0.0.1:8000'

def word_is_english(word):
   for c in word:
      if 'a' <= c <= 'z' or 'A' <= c <= 'C':
         return True
   return False

def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()

def handle_community(community, umls_data, heb_searcher, eng_searcher):
    calculated_relatedness_dict = get_relatedness_dict(community)

    print(f"community: {community}")
    comm_df = pd.read_excel(input_dir + os.sep + community + ".xlsx")

    all_relatedness = []
    for row_idx, row in comm_df.iterrows():
        match = row['match']

        word_and_lemma = get_word_token_lemma(match, row)

        english_case = word_is_english(match)

        if english_case:
            searcher = eng_searcher
        else:
            searcher = heb_searcher

        if word_and_lemma:
            search_results_word = searcher.ranked_search(word_and_lemma[0], similarity_threshold)[:3]
            search_results_lemma = searcher.ranked_search(word_and_lemma[1], similarity_threshold)[:3]
            search_results = search_results_word + search_results_lemma
        else:
            search_results = []

        if search_results != []:
            highest_relatedness_option, highest_relatedness_option_value = get_relatedness_data(calculated_relatedness_dict, search_results, umls_data, english_case)
        else:
            highest_relatedness_option, highest_relatedness_option_value = None, -1

        # if row_idx % 100 == 0:
        #     print(f"match: {row['match']}, word_and_lemma: {word_and_lemma} highest_relatedness_option: {highest_relatedness_option}, highest_relatedness_option_value: {highest_relatedness_option_value}")

        all_relatedness.append(highest_relatedness_option_value)

    comm_df['relatedness'] = all_relatedness

    comm_df.to_excel(output_dir + os.sep + community + "_output.xlsx", index=False)


def get_word_token_lemma(match, row):
    prefixes = ["ב", "ה", "ול", "ל", "כש", "שה", "ו", "מ"]
    match_has_pref = False
    for pref in prefixes:
        if match.startswith(pref):
            match_has_pref = True
            break
    if match_has_pref:
        most_similar_word_and_lemma = get_seperated_word(match, row)
    else:
        most_similar_word_and_lemma = match, match
    return most_similar_word_and_lemma


def get_seperated_word(match, row):
    tokenized_txt = row['tokenized_txt']
    if '\xa0' in tokenized_txt:
        tokenized_txt = tokenized_txt.replace('\xa0', "")
    words_and_lemmas = get_words_and_lemmas(tokenized_txt)
    all_words = [w.split(" ")[0] for w in words_and_lemmas]
    all_lemmas = [w.split(" ")[1] for w in words_and_lemmas]
    most_similar_word_and_lemma = None
    most_similar_w_value = 0
    for w, lemma in zip(all_words, all_lemmas):
        sim = max(words_similarity(match, w), words_similarity(match, lemma))
        if sim > 0.8 and sim > most_similar_w_value:
            most_similar_word_and_lemma = w, lemma
            most_similar_w_value = sim
    return most_similar_word_and_lemma

def get_relatedness_data(calculated_relatedness_dict, search_results, umls_data, english_case):
    highest_relatedness_option = None
    highest_relatedness_option_value = 0
    for option in search_results:
        cosine_sim, word_match_in_umls = option
        match_relatedness, match_str = get_match_info(calculated_relatedness_dict, english_case, umls_data,
                                                      word_match_in_umls)
        if match_relatedness > highest_relatedness_option_value:
            highest_relatedness_option_value = match_relatedness
            highest_relatedness_option = match_str
    return highest_relatedness_option, highest_relatedness_option_value


def get_match_info(calculated_relatedness_dict, english_case, umls_data, word_match_in_umls):
    if english_case:
        match_relatedness, match_str = geb_eng_match_info(calculated_relatedness_dict, word_match_in_umls)
    else:
        match_relatedness, match_str = get_heb_match_info(calculated_relatedness_dict, umls_data, word_match_in_umls)
    return match_relatedness, match_str


def geb_eng_match_info(calculated_relatedness_dict, word_match_in_umls):
    match_str = word_match_in_umls
    if word_match_in_umls in calculated_relatedness_dict:
        match_relatedness = float(calculated_relatedness_dict[word_match_in_umls])
    elif word_match_in_umls.lower() in calculated_relatedness_dict:
        match_relatedness = float(calculated_relatedness_dict[word_match_in_umls.lower()])
    else:
        match_relatedness = -1
    return match_relatedness, match_str


def get_heb_match_info(calculated_relatedness_dict, umls_data, word_match_in_umls):
    matched_rows = umls_data.loc[word_match_in_umls]
    if type(matched_rows) == pd.Series:
        match_str = matched_rows['STR']
        if match_str in calculated_relatedness_dict:
            match_relatedness = float(calculated_relatedness_dict[match_str])
        else:
            lower_match = match_str.lower()
            if lower_match in calculated_relatedness_dict:
                match_relatedness = float(calculated_relatedness_dict[lower_match])
            else:
                match_relatedness = -1
    else:
        matched_eng_values = matched_rows['STR'].values
        best_eng_option, best_eng_option_relatedness_value = get_best_eng_option(calculated_relatedness_dict,
                                                                                 matched_eng_values)
        match_str = best_eng_option
        match_relatedness = best_eng_option_relatedness_value
    return match_relatedness, match_str


def get_best_eng_option(calculated_relatedness_dict, matched_eng_values):
    best_eng_option = None
    best_eng_option_relatedness_value = -1
    for eng_option in matched_eng_values:
        if eng_option in calculated_relatedness_dict:
            eng_option_relatedness = float(calculated_relatedness_dict[eng_option])
        else:
            eng_option_relatedness = -1
        if eng_option_relatedness > best_eng_option_relatedness_value:
            best_eng_option_relatedness_value = eng_option_relatedness
            best_eng_option = eng_option
    return best_eng_option, best_eng_option_relatedness_value


def get_umls_data():
    umls_data = pd.read_csv(umls_data_path)
    print(f"Got UMLS data at length {len(umls_data)}")
    umls_data = umls_data[~umls_data['STR'].isna()]

    umls_data.sort_values(by=['HEB'], inplace=True)
    umls_data.reset_index(inplace=True)


    heb_umls_list = list(umls_data['HEB'].values)
    eng_umls_list = list(umls_data['STR'].values)

    heb_db = DictDatabase(CharacterNgramFeatureExtractor(2))
    eng_db = DictDatabase(CharacterNgramFeatureExtractor(2))

    for heb_w in heb_umls_list:
        heb_db.add(heb_w)

    for eng_w in eng_umls_list:
        eng_db.add(eng_w)
    umls_data.set_index('HEB', inplace=True)
    return umls_data, heb_db, eng_db

def get_relatedness_dict(community):
    with open(calculated_relatedness_dir + os.sep + community + "_relatedness.txt", encoding='utf-8') as f:
        lines = [x.rstrip('\n') for x in f.readlines()]
    calculated_relatedness_dict = {}
    for l in lines:
        relatedness = l.split("<>")[0]
        term = l.split("<>")[-1].split("(")[0]
        calculated_relatedness_dict[term] = relatedness
    return calculated_relatedness_dict

def get_words_and_lemmas(post_example):
    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(post_example, ip)
    dataframes = get_dataframes_of_words_and_lemmas(md_lattice)
    if dataframes == []:
        return []

    words_and_lemmas = get_all_words_and_lemmas_from_df(dataframes)
    return words_and_lemmas

def get_dataframes_of_words_and_lemmas(md_lattice):
    if 'num_last' not in md_lattice:
        return []
    md_lattice['num_last'] = md_lattice['num_last'].apply(lambda x: int(x))
    indexes_of_starts = find_indexes_of_start(md_lattice)
    if len(indexes_of_starts) == 1:
        dataframes = [md_lattice]
    else:
        dataframes = build_dataframes(indexes_of_starts, md_lattice)
    return dataframes

def build_dataframes(indexes_of_starts, md_lattice):
    pairs = list(zip(indexes_of_starts, indexes_of_starts[1:]))
    dataframes = []
    for pair in pairs:
        df = md_lattice.iloc[pair[0]:pair[1]]
        dataframes.append(df)
    last_df = md_lattice.iloc[pairs[-1][1]:]
    dataframes.append(last_df)
    return dataframes

def find_indexes_of_start(md_lattice):
    indexes_of_starts = []
    for idx, val in enumerate(md_lattice['num_last'].values):
        if val == 1:
            indexes_of_starts.append(idx)
    return indexes_of_starts


def max_len(s):
    return max(s, key=len)

def get_all_words_and_lemmas_from_df(dataframes):
    all_words_and_lemmas = []
    for df in dataframes:
        words_and_lemmas_groups = df.groupby('num_last').agg({'lemma': max_len, 'word': max_len})
        words_and_lemmas_groups = words_and_lemmas_groups.sort_values(by=['num_last'])
        words_and_lemmas = [x['lemma'] + " " + x['word'] for x in
                            words_and_lemmas_groups.to_dict('index').values()]
        all_words_and_lemmas += words_and_lemmas
    return all_words_and_lemmas

if __name__ == '__main__':
    umls_data, heb_db, eng_db = get_umls_data()
    heb_searcher = Searcher(heb_db, CosineMeasure())
    eng_searcher = Searcher(eng_db, CosineMeasure())

    handle_community("diabetes", umls_data, heb_searcher, eng_searcher)
    handle_community("sclerosis", umls_data, heb_searcher, eng_searcher)
    handle_community("depression", umls_data, heb_searcher, eng_searcher)

    print("Done")