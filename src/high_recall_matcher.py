import difflib
import json
import os
import time

import pandas as pd
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

stopword_path = r"E:\mdtel_data\data\high_recall_matcher\heb_stop_words.txt"
yap_processed_dir = r"E:\mdtel_data\data\high_recall_matcher\posts\yap_processed"
umls_df_data_path = r"E:\mdtel_data\data\high_recall_matcher\HEB_TO_ENG_MRCONSO_RELATEDNESS.csv"
output_dir = r"E:\mdtel_data\data\high_recall_matcher\output"

SINGLE_WORD_SIMILARITY_THRESHOLD = 0.92
MULTI_WORD_SIMILARITY_THRESHOLD = 0.92

DIABETES, SCLEROSIS, DEPRESSION = 'diabetes', 'sclerosis', 'depression'

with open(stopword_path, encoding='utf-8') as f:
    heb_stop_words = [x.rstrip('\n') for x in f.readlines()]

NUMBER_OF_GRAMS = 3

DEBUG = True

STRING_COLUMN = 'STR'

if DEBUG:
    print(f"*** DEBUG MODE: Taking 100 posts only ***")
else:
    print(f"Debug == False. Might take some time.")


def get_data(chosen_community):
    community_dir = yap_processed_dir + os.sep + chosen_community
    all_community_msgs_data = []
    for idx, msg_file in enumerate(os.listdir(community_dir)):
        with open(community_dir + os.sep + msg_file, encoding='utf-8') as f:
            msg_data = json.load(f)
        del msg_data['segmented_text']
        del msg_data['words_and_lemmas']
        all_community_msgs_data.append(msg_data)
    if DEBUG:
        print(f"DEBUG. Taking sub sample of community data: {len(all_community_msgs_data)}")
        all_community_msgs_data = all_community_msgs_data[:105]
    return all_community_msgs_data

def word_is_english(word):
   for c in word:
      if 'a' <= c <= 'z' or 'A' <= c <= 'C':
         return True
   return False

def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


def is_good_match(sim, word_match, i, similarity_threshold, all_matches_found):
    return sim > similarity_threshold and i == len(word_match.split(" ")) and sim > similarity_threshold and word_match not in [m[0] for m in all_matches_found]

def find_umls_match_fast(msg_txt, searcher):
    msg_words = msg_txt.split(" ")
    msg_words = [w for w in msg_words if w != "" and w != " " and len(w) > 2]
    ngrams = list(zip(*[msg_words[i:] for i in range(NUMBER_OF_GRAMS)]))

    all_matches_found = []

    for gram in ngrams:
        for i in range(1, NUMBER_OF_GRAMS + 1):
            similarity_threshold = SINGLE_WORD_SIMILARITY_THRESHOLD if i == 1 else MULTI_WORD_SIMILARITY_THRESHOLD
            cand_term = " ".join(gram[:i])

            search_result = searcher.ranked_search(cand_term, similarity_threshold)
            if search_result != []:
                cosine_sim, word_match = search_result[0]  # Cosine-Sim. I can demand another sim
                sim = words_similarity(word_match, cand_term)
                if is_good_match(sim, word_match, i, similarity_threshold, all_matches_found):
                    result = (cand_term, word_match, round(sim, 3))
                    all_matches_found.append(result)

    return set(all_matches_found)


def main():
    heb_db, eng_db, umls_data = get_umls_data()

    heb_searcher = Searcher(heb_db, CosineMeasure())
    eng_searcher = Searcher(eng_db, CosineMeasure())

    handle_community(DIABETES, heb_searcher, eng_searcher, umls_data)
    handle_community(SCLEROSIS, heb_searcher, eng_searcher, umls_data)
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
        eng_db.add(eng_w)

    return heb_db, eng_db, umls_data


def handle_community(chosen_community, heb_searcher, eng_searcher, umls_data):
    print(f"Handling community {chosen_community}")
    all_community_msgs_data = get_data(chosen_community)

    all_tokenized_txts = []
    all_matches_found = []
    start_time = time.time()

    for idx, msg_data in enumerate(all_community_msgs_data):
        msg_eng_text = get_eng_text(msg_data)

        english_matches_found = find_umls_match_fast(msg_eng_text, eng_searcher)
        if english_matches_found != set():
            english_matches_found = add_eng_umls_data(english_matches_found, umls_data)

        matches_found_tokenized_text = find_umls_match_fast(msg_data['tokenized_text'], heb_searcher)
        matches_found_lemmas = find_umls_match_fast(msg_data['lemmas'], heb_searcher)

        heb_matches_found = matches_found_tokenized_text.union(matches_found_lemmas)
        if heb_matches_found != ():
            heb_matches_found = take_highest_sim_matches(heb_matches_found, umls_data)
            heb_matches_found = add_heb_umls_data(heb_matches_found, umls_data)

        msg_matches_found = list(heb_matches_found.union(english_matches_found))

        all_tokenized_txts.append(msg_data['tokenized_text'])
        all_matches_found.append(json.dumps(msg_matches_found, ensure_ascii=False))

        print_stats(all_community_msgs_data, idx, msg_data, msg_matches_found, start_time)

    detection_df = pd.DataFrame()
    detection_df['tokenized_txt'] = all_tokenized_txts
    detection_df['matches_found'] = all_matches_found

    if DEBUG:
        output_path = output_dir + os.sep + chosen_community + "_debug.xlsx"
    else:
        output_path = output_dir + os.sep + chosen_community + ".xlsx"

    detection_df.to_excel(output_path, encoding='utf-8', index=False)

    print(f"Finished with community {chosen_community}")
    print(f"Wrote to path {output_path}")


def print_stats(all_community_msgs_data, idx, msg_data, msg_matches_found, start_time):
    if idx % 100 == 0 and idx > 0:
        print(f"idx: {idx}, out of: {len(all_community_msgs_data)}, total {round((idx / len(all_community_msgs_data)) * 100, 3)}%")
        elapsed_time = round(time.time() - start_time, 3)
        msgs_left = len(all_community_msgs_data) - idx
        time_left = round((msgs_left / idx) * elapsed_time, 3)
        print(f"elapsed time: {elapsed_time}, estimated time left: {time_left}s")
    if DEBUG or idx % 100 == 0:
        if msg_matches_found != None and msg_matches_found != []:
            print(msg_data['tokenized_text'])
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


def add_heb_umls_data(heb_matches, umls_data):
    heb_matches_with_codes = []
    for match in heb_matches:
        cand, word_match, sim = match
        idx_of_match_in_df = umls_data['HEB'].searchsorted(word_match)
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        match_eng = umls_data.iloc[idx_of_match_in_df][STRING_COLUMN]
        heb_matches_with_codes.append((cand + "-" + word_match + " ( " + match_eng + " )" + " : " + str(sim), match_cui))
    return set(heb_matches_with_codes)

def add_eng_umls_data(eng_matches, umls_data):
    eng_matches_with_codes = []
    for match in eng_matches:
        cand, word_match, sim = match
        idx_of_match_in_df = umls_data[STRING_COLUMN][umls_data[STRING_COLUMN] == word_match].index[0]
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        eng_matches_with_codes.append((cand + "-" + word_match + " : " + str(sim), match_cui))
    return set(eng_matches_with_codes)

def get_eng_text(msg_data):
    msg_words = msg_data['tokenized_text'].split(" ")
    msg_eng_text = " ".join([w.lower() for w in msg_words if w != "" and w != " " and len(w) > 2 and word_is_english(w)])
    return msg_eng_text

if __name__ == '__main__':
    main()