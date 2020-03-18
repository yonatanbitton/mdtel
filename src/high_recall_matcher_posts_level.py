import difflib
import json
import os
import re
import sys

import pandas as pd
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

module_path = os.path.abspath(os.path.join('..', os.getcwd()))
sys.path.append(module_path)

from config import *

stopword_path = data_dir + r"high_recall_matcher\heb_stop_words.txt"
input_dir = data_dir + r"high_recall_matcher\posts\lemlda"

umls_df_data_path = data_dir + r"high_recall_matcher\HEB_TO_ENG_DISORDERS_CHEMICALS_utf-8-sig.csv"

output_dir = data_dir + r"high_recall_matcher\output"

LOW_SINGLE_WORD_SIMILARITY_THRESHOLD = 0.80
UP_SINGLE_WORD_SIMILARITY_THRESHOLD = 0.85
LOW_MULTI_WORD_SIMILARITY_THRESHOLD = 0.85
UP_MULTI_WORD_SIMILARITY_THRESHOLD = 0.90

DIABETES, SCLEROSIS, DEPRESSION = 'diabetes', 'sclerosis', 'depression'

with open(stopword_path, encoding='utf-8') as f:
    heb_stop_words = [x.rstrip('\n') for x in f.readlines()]

NUMBER_OF_GRAMS = 3

STRING_COLUMN = 'STR'

if DEBUG:
    print(f"*** DEBUG MODE: Taking 20 posts only ***")
else:
    print(f"Debug == False. Might take some time.")

number_of_not_exact_matches = 0
number_of_lemma_matches = 0


def word_is_english(word):
   for c in word:
      if 'a' <= c <= 'z' or 'A' <= c <= 'C':
         return True
   return False

def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


def is_good_match(sim, word_match, i, similarity_threshold):
    return len(word_match) > 3 and sim > similarity_threshold and i == len(word_match.split(" "))

# def is_good_match(sim, word_match, i, similarity_threshold, all_matches_found):
#     return len(word_match) > 3 and sim > similarity_threshold and i == len(word_match.split(" ")) and word_match not in [m[0] for m in all_matches_found]

def find_umls_match_fast(msg_txt, searcher, row, msg_key_lang):
    if msg_txt == "":
        return []

    ngrams = prepare_msg_ngrams(msg_txt)

    all_matches_found = []

    for gram in ngrams:
        for i in range(1, NUMBER_OF_GRAMS + 1):
            low_similarity_threshold = LOW_SINGLE_WORD_SIMILARITY_THRESHOLD if i == 1 else LOW_MULTI_WORD_SIMILARITY_THRESHOLD
            up_similarity_threshold = UP_SINGLE_WORD_SIMILARITY_THRESHOLD if i == 1 else UP_MULTI_WORD_SIMILARITY_THRESHOLD
            cand_term = " ".join(gram[:i])
            search_result = searcher.ranked_search(cand_term, low_similarity_threshold)
            if search_result != []:
                cosine_sim, umls_match = search_result[0]  # Cosine-Sim. I can demand another sim
                sim = words_similarity(umls_match, cand_term)
                if is_good_match(sim, umls_match, i, up_similarity_threshold):
                    all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset = get_cand_occ_in_text(
                        all_matches_found, cand_term, msg_key_lang, row)
                    result = {'cand_match': cand_term, 'umls_match': umls_match, 'cand_match_occurence_in_txt': cand_match_occurence_in_txt,
                              'sim': round(sim, 3), 'curr_occurence_offset': curr_occurence_offset,
                              'all_match_occ': all_match_occ, 'msg_key_lang': msg_key_lang}
                    all_matches_found.append(result)

    return all_matches_found


def get_cand_occ_in_text(all_matches_found, cand_term, msg_key_lang, row):
    cand_match_occurence_in_txt = cand_term
    if msg_key_lang == 'lemmas':
        curr_occurence_offset, all_match_occ = "lemma", "lemma"
    else:
        curr_occurence_offset, all_match_occ = find_occurence_offset(all_matches_found, cand_match_occurence_in_txt,
                                                                     row, msg_key_lang)
    return all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset


def find_occurence_offset(all_matches_found, cand_match_occurence_in_txt, row, msg_key_lang):
    post_txt = row['post_txt']
    post_txt = post_txt.replace("\n", " ").replace(".", " ").replace(","," ")
    # all_match_occ = [m.start() for m in re.finditer(r"." + cand_match_occurence_in_txt, post_txt)]
    all_match_occ = [m.start() for m in re.finditer(cand_match_occurence_in_txt, post_txt)]

    curr_occurence = 0
    for m in all_matches_found:
        if m['cand_match_occurence_in_txt'] == cand_match_occurence_in_txt:
            curr_occurence += 1

    if all_match_occ == [] or len(all_match_occ) <= curr_occurence:
        global number_of_not_exact_matches
        number_of_not_exact_matches += 1
        return None, None

    curr_occurence_offset = all_match_occ[curr_occurence]
    return curr_occurence_offset, all_match_occ


def prepare_msg_ngrams(msg_txt):
    msg_txt = msg_txt.replace(".", " ")
    msg_words = msg_txt.split(" ")
    msg_words = [w for w in msg_words if w != "" and w != " " and len(w) >= 2]
    if len(msg_words) < 3:
        msg_words.append("PAD")
    ngrams = list(zip(*[msg_words[i:] for i in range(NUMBER_OF_GRAMS)]))
    if len(ngrams) > 0:
        last_gram = ngrams[-1]
        extra_gram = last_gram[1], last_gram[2], 'PAD'
        ngrams.append(extra_gram)
        extra_gram_2 = last_gram[2], 'PAD', 'PAD'
        ngrams.append(extra_gram_2)
    return ngrams


def main():
    heb_db, eng_db, umls_data = get_umls_data()

    heb_searcher = Searcher(heb_db, CosineMeasure())
    eng_searcher = Searcher(eng_db, CosineMeasure())

    # handle_community(SCLEROSIS, heb_searcher, eng_searcher, umls_data)
    # handle_community(DIABETES, heb_searcher, eng_searcher, umls_data)
    handle_community(DEPRESSION, heb_searcher, eng_searcher, umls_data)

    print("Done")

def try_to_json_loads_or_none(x):
    try:
        return json.loads(x)
    except Exception as ex:
        return None

def get_words(term):
    return [w for w in term.split(" ") if w != ""]

def english_term_is_good(x):
    first_creteria = len(x) > 3
    if not first_creteria:
        return False
    bad_w = False
    for w in x.split(" "):
        if len(w) == 1:
            bad_w = True
            break
    if bad_w:
        return False
    else:
        return True

def get_umls_data():
    umls_data = pd.read_csv(umls_df_data_path)
    print(f"Got UMLS data at length {len(umls_data)}")
    umls_data = umls_data[~umls_data['STR'].isna()]

    # if DEBUG:
    #     umls_data = umls_data.head(10000)

    umls_data.sort_values(by=['HEB'], inplace=True)
    umls_data = umls_data[umls_data['STR'].apply(lambda x: len(x) > 4)]
    umls_data = umls_data[umls_data['HEB'].apply(lambda x: len(x) > 3)]
    umls_data.reset_index(inplace=True)


    heb_umls_list = list(umls_data['HEB'].values)
    eng_umls_list = list(umls_data[STRING_COLUMN].values)

    heb_db = DictDatabase(CharacterNgramFeatureExtractor(2))
    eng_db = DictDatabase(CharacterNgramFeatureExtractor(2))

    for heb_w in heb_umls_list:
        heb_db.add(heb_w)

    for eng_w in eng_umls_list:
        lower_eng_w = eng_w.lower()
        eng_db.add(lower_eng_w)

    return heb_db, eng_db, umls_data


def get_data(chosen_community):
    community_df = pd.read_excel(input_dir + os.sep + chosen_community + "_posts.xlsx")
    community_df['words_and_lemmas'] = community_df['words_and_lemmas'].apply(lambda x: json.loads(x) if str(x) != 'nan' else 'nan')
    # if DEBUG:
    #     community_df = community_df.head(20)

    # Taking only posts that have labels (Not predicting posts without labels to measure performance)
    community_df = take_only_posts_which_has_labels(chosen_community, community_df)

    return community_df


def take_only_posts_which_has_labels(chosen_community, community_df):
    labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'
    labels_df = pd.read_csv(labels_dir + os.sep + chosen_community + "_labels.csv")
    all_labeled_texts = [x.strip() for x in list(labels_df['text'].values)]
    # all_labeled_tokenized_texts = list(labels_df['tokenized_text'].values)
    number_of_posts_needed = {'sclerosis': 265, 'diabetes': 266, 'depression': 271}
    community_df = community_df[community_df['post_txt'].apply(lambda x: x.strip() in all_labeled_texts)]
    print(f"Number of posts: {len(community_df)}, needed: {number_of_posts_needed[chosen_community]}")
    assert abs(len(community_df) - number_of_posts_needed[chosen_community]) < 3  # Can accept small difference
    return community_df


def handle_community(chosen_community, heb_searcher, eng_searcher, umls_data):
    print(f"Handling community {chosen_community}")
    community_df = get_data(chosen_community)
    all_matches_found = []

    number_of_posts = len(community_df)

    for row_idx, row in community_df.iterrows():

        msg_matches_found = get_english_and_hebrew_matches(eng_searcher, heb_searcher, row, umls_data)

        all_matches_found.append(json.dumps(msg_matches_found, ensure_ascii=False))

        print_stats(row_idx, row, msg_matches_found, number_of_posts)

    community_df['matches_found'] = all_matches_found

    if DEBUG:
        output_path = output_dir + os.sep + chosen_community + "_debug.csv"
    else:
        output_path = output_dir + os.sep + chosen_community + ".csv"

    community_df.to_csv(output_path, encoding='utf-8-sig', index=False)

    print(f"Finished with community {chosen_community}")
    print(f"Wrote to path {output_path}")


def get_english_and_hebrew_matches(eng_searcher, heb_searcher, row, umls_data):
    if str(row['tokenized_text']) == 'nan':
        return []
    english_matches_found = get_english_matches(eng_searcher, row, umls_data)
    all_hebrew_matches_found = get_hebrew_matches(heb_searcher, row, umls_data)
    # msg_matches_found = list(all_hebrew_matches_found.union(english_matches_found))

    msg_matches_found = list_union(all_hebrew_matches_found, english_matches_found, post_key='union')

    return msg_matches_found


def get_hebrew_matches(heb_searcher, row, umls_data):
    all_hebrew_matches_found = []
    # for hebrew_key in ['tokenized_text', 'segmented_text', 'lemmas']:
    for hebrew_key in ['tokenized_text', 'lemmas']:
        msg_key_txt = row[hebrew_key]
        matches_found_for_key = find_umls_match_fast(msg_key_txt, heb_searcher, row, hebrew_key)

        if matches_found_for_key not in [(), []]:
            # matches_found_for_key = take_highest_sim_matches(matches_found_for_key, umls_data)
            matches_found_for_key = add_heb_umls_data(matches_found_for_key, umls_data, hebrew_key)

        all_hebrew_matches_found = list_union(all_hebrew_matches_found, matches_found_for_key, hebrew_key)

    return all_hebrew_matches_found

def get_english_matches(eng_searcher, row, umls_data):
    msg_eng_text = get_eng_text(row['tokenized_text'])
    english_matches_found = find_umls_match_fast(msg_eng_text, eng_searcher, row, msg_key_lang='english')
    if english_matches_found not in [set(), []]:
        english_matches_found = add_eng_umls_data(english_matches_found, umls_data, row)
    return english_matches_found


def list_union(all_hebrew_matches_found, matches_found_for_key, post_key):
    for d in matches_found_for_key:
        all_existing_umls_matches = [m['umls_match'] for m in all_hebrew_matches_found]
        if d['umls_match'] not in all_existing_umls_matches:
            if post_key == 'lemmas':
                global number_of_lemma_matches
                number_of_lemma_matches += 1
            all_hebrew_matches_found.append(d)
    return all_hebrew_matches_found


def print_stats(idx, row, msg_matches_found, number_of_posts):
    if idx % 100 == 0 and idx > 0:
        print(f"idx: {idx}, out of: {number_of_posts}, total {round((idx / number_of_posts) * 100, 3)}%, number_of_lemma_matches: {number_of_lemma_matches}")
    if DEBUG or idx % 100 == 0:
        if msg_matches_found != None and msg_matches_found != []:
            print(row['tokenized_text'])
            print(msg_matches_found)
            print()


def take_highest_sim_matches(heb_matches_found, umls_data):
    new_heb_matches = []
    sorted_heb_matches_found = sorted(list(heb_matches_found), key=lambda x: x[-1], reverse=True)
    for t in sorted_heb_matches_found:
        match_heb_term = t[1]
        if not match_heb_term in [l[1] for l in new_heb_matches]:
            new_heb_matches.append(t)
    heb_matches_found = new_heb_matches
    return heb_matches_found


def get_semantic_type(match_tui):
    if match_tui in disorder_tuis:
        return DISORDER
    elif match_tui in chemical_or_drug_tuis:
        return CHEMICAL_OR_DRUG


def add_heb_umls_data(heb_matches, umls_data, hebrew_key):
    heb_matches_with_codes = []
    for match_d in heb_matches:
        idx_of_match_in_df = umls_data['HEB'].searchsorted(match_d['umls_match'])
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        match_tui = umls_data.iloc[idx_of_match_in_df]['TUI']
        semantic_type = get_semantic_type(match_tui)
        match_eng = umls_data.iloc[idx_of_match_in_df][STRING_COLUMN]
        heb_d = {'cand_match': match_d['cand_match'], 'umls_match': match_d['umls_match'], 'sim': match_d['sim'],
                 'cui': match_cui, 'match_eng': match_eng, 'hebrew_key': hebrew_key, 'match_tui': match_tui,
                 'semantic_type': semantic_type, 'all_match_occ': match_d['all_match_occ'],
                 'curr_occurence_offset': match_d['curr_occurence_offset']}
        if heb_d not in heb_matches_with_codes:
            heb_matches_with_codes.append(heb_d)
        # heb_matches_with_codes.append((cand + "-" + word_match + " ( " + match_eng + " )" + " : " + str(sim), match_cui))
    return heb_matches_with_codes

def add_eng_umls_data(eng_matches, umls_data, row):
    eng_matches_with_codes = []
    for match_d in eng_matches:
        idx_of_match_in_df = umls_data[umls_data[STRING_COLUMN].apply(lambda x: x.lower() == match_d['umls_match'])].index[0]
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        match_tui = umls_data.iloc[idx_of_match_in_df]['TUI']
        semantic_type = get_semantic_type(match_tui)
        eng_d = {'cand_match': match_d['cand_match'], 'umls_match': match_d['umls_match'], 'sim': match_d['sim'],
                 'cui': match_cui, 'match_tui': match_tui, 'semantic_type': semantic_type, 'all_match_occ': match_d['all_match_occ'],
                 'curr_occurence_offset': match_d['curr_occurence_offset']}
        if eng_d not in eng_matches_with_codes:
            eng_matches_with_codes.append(eng_d)
    return eng_matches_with_codes

def get_eng_text(msg_tokenized_txt):
    msg_words = msg_tokenized_txt.split(" ")
    msg_eng_text = " ".join([w.lower() for w in msg_words if w != "" and w != " " and len(w) > 2 and word_is_english(w)])
    return msg_eng_text

if __name__ == '__main__':
    main()