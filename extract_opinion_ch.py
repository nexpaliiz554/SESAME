import os
import pandas as pd
import json
from itertools import chain
import numpy as np
from keras.api._v2.keras import Model
from pycorenlp import StanfordCoreNLP as pyCoreNLP

import EASTER_ch
from config import *
from EASTER_ch import DataGenerator
from shap.maskers import Text
from shap import Explainer

import nltk

std_p_value = 0.8
is_consider_multi_sentiment_through_adversative = True

nlp = pyCoreNLP('http://localhost:9876')
# 1. Copy StanfordCoreNLP-chinese.properties into STANFORD_CORE_NLP_PATH
# 2. Navigate to the STANFORD_CORE_NLP_PATH directory
# 3. Run the following command: java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9876 -timeout 15000
class OpinionExtractor:
    def __init__(self, target_name: str=None, shap_file: str=None):
        phone_pos_optimization = f'{BASE_DIR}/data/pos_analysis_optimization_results/phone_tagged.txt'
        camera_pos_optimization = f'{BASE_DIR}/data/pos_analysis_optimization_results/camera_tagged.txt'
        pos_optimization_map = {"phone": phone_pos_optimization,
                                "camera": camera_pos_optimization}
        self.tagged_file = ""
        self.shap_file = shap_file
        self.words_lead_to_clause = ["，"]
        self.pos_tags_lead_to_clause = ["CC"]
        self.clause_label_list = ['IP']
        self.pos_map = {
            "NR": "n", "NT": "n", "NN": "n",
            "VC": "v", "VE": "v", "VV": "v",
            "JJ": "a", "VA": "a",
            "AD": "r",
        }
        # load sentiwordnet
        with open(f'{BASE_DIR}/data/dict/台湾大学NTUSD简体中文情感词典/NTUSD_negative_simplified.txt', 'r', encoding='utf-16') as f:
            senti_words_1 = [line.strip() for line in f if line.strip()]
        with open(f'{BASE_DIR}/data/dict/台湾大学NTUSD简体中文情感词典/NTUSD_positive_simplified.txt', 'r', encoding='utf-16') as f:
            senti_words_2 = [line.strip() for line in f if line.strip()]
        self.sentiwordnet = senti_words_1 + senti_words_2
        self.data_list = []
        if target_name != None:
            self.tagged_file = pos_optimization_map[target_name]
        if shap_file != None:
            self.data_list = self._load_data()

    def get_texts(self):
        return [data['sentence'] for data in self.data_list]

    def _is_potential_aspect(self, word: str, pos: str):
        """
        To determine if it is a potential aspect,
        the conditions for being a potential aspect are:
            1) The word is not in sentiwordnet ,
            2) It is a noun.
        """
        return (not self.sentiwordnet.__contains__(word)) and self.pos_map[pos]=="n"

    def extract_top_k_opinion(self, k: int):
        """
        Extract the top k highest shap value as opinion
        """
        opinion_list = []
        for data in self.data_list:
            senti = data['sentiment']
            one_text_opinion_list = []
            senti_SHAP_index = senti + 4
            words_info = sorted(filter(lambda x: self.pos_map.get(x[2], None) is not None, data['words_info']),key=lambda x: x[senti_SHAP_index], reverse=True)
            if len(words_info) == 0:
                opinion = {'sentiment': senti,
                           'actual_index': [-1, -1],
                           'shap_value': -1,
                           'is_potential_aspect': False}
                one_text_opinion_list.append( opinion )
            else:
                for x in words_info[0:k]:
                    opinion = {'sentiment': senti,
                               'actual_index': x[6],
                               'shap_value': x[senti_SHAP_index],
                               'is_potential_aspect': self._is_potential_aspect(word=x[1],pos=x[2])}
                    one_text_opinion_list.append( opinion )

            opinion_list.append( one_text_opinion_list )
        return opinion_list

    def _get_clause_with_constituency(self, data):
        '''
        Split text based on constituency analysis
        '''
        if len(data['words_info']) == 0:
            return data['words_info']
        constituency_tree_str = data['constituency']
        word_list = [ x[1] for x in data['words_info'] ]
        start_index = 0
        for i in range(len(word_list)):
            if start_index >= len(constituency_tree_str):
                break
            word = "-LRB-" if word_list[i] == "(" else "-RRB-" if word_list[i] == ")" else word_list[i]
            head_str = constituency_tree_str[:start_index]
            next_rpare_index = constituency_tree_str.find(")",constituency_tree_str.find("(",start_index))
            middle_str = constituency_tree_str[start_index:next_rpare_index+1]
            tail_str = constituency_tree_str[next_rpare_index+1:]
            middle_str = middle_str.replace(f"{word})",f"{i})",1)
            constituency_tree_str = head_str + middle_str + tail_str
            start_index = len(head_str + middle_str)
        tree = nltk.Tree.fromstring(constituency_tree_str)
        clause_start_index_set = set()
        for subtree in tree.subtrees():
            try:
                start_index = int(subtree.leaves()[0])
            except ValueError:
                continue
            if subtree.label() in self.clause_label_list:
                clause_start_index_set.add(start_index)
        clauses = []
        current_clause = []
        start_index_set = clause_start_index_set
        for i in range(len(data['words_info'])):
            item = data['words_info'][i]
            if (i in start_index_set) \
                    or (item[1] in self.pos_tags_lead_to_clause) \
                    or (item[0] in self.words_lead_to_clause):
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            current_clause.append(item)
        if current_clause:
            clauses.append(current_clause)
        return clauses

    def extract_clause_opinion_for_text_given_senti(self, senti: int, clause_words_info_list: list, data: list, std_p: float, preset_standard: float = None):
        is_pure_opinion = False
        opinion_list_for_text = []

        senti_SHAP_index = senti + 4
        shap_values = [x[senti_SHAP_index] for x in data['words_info']]
        if len(data['words_info']) != 0:
            standard = np.mean(shap_values) + std_p * np.std(shap_values)
        else:
            standard = -1
        if preset_standard is not None:
            standard = preset_standard

        max_clause_opinion = None
        for index, clause_words_info in enumerate(clause_words_info_list):
            clause_words_info = sorted(filter(lambda x: self.pos_map.get(x[2], None) is not None, clause_words_info),
                                       key=lambda x: x[senti_SHAP_index], reverse=True)
            if len(clause_words_info) == 0:
                continue
            clause_opinion = clause_words_info[0]
            if max_clause_opinion is None or clause_opinion[senti_SHAP_index] > max_clause_opinion[senti_SHAP_index]:
                max_clause_opinion = clause_opinion
            if clause_opinion[senti_SHAP_index] < standard:
                continue
            opinion = {'clause_index': index,
                       'sentiment': senti,
                       'actual_index': clause_opinion[6],
                       'shap_value': clause_opinion[senti_SHAP_index],
                       'is_potential_aspect': self._is_potential_aspect(word=clause_opinion[1], pos=clause_opinion[2])}
            opinion_list_for_text.append(opinion)

        if len(opinion_list_for_text) == 0:
            if max_clause_opinion is not None:
                opinion = {'clause_index': -1,
                           'sentiment': senti,
                           'actual_index': max_clause_opinion[6],
                           'shap_value': max_clause_opinion[senti_SHAP_index],
                           'is_potential_aspect': self._is_potential_aspect(word=max_clause_opinion[1],
                                                                            pos=max_clause_opinion[2])}
            else:
                opinion = {'clause_index': -1,
                           'sentiment': senti,
                           'actual_index': [-1, -1],
                           'shap_value': -1,
                           'is_potential_aspect': False}
            is_pure_opinion = True
            opinion_list_for_text.append(opinion)

        is_pure_opinion = is_pure_opinion or len(clause_words_info_list) == 1

        return is_pure_opinion, standard, opinion_list_for_text

    def may_contain_multi_sentiment(self,data):
        adversative_list = ["但是"]
        lemma_list = [x[1] for x in data['words_info']]
        for idx, word in enumerate(lemma_list):
            # If "但是" is at the beginning of the sentence, there may not necessarily be two sentiments in the sentence
            if idx == 0 and (word in ["但是"]) :
                continue
            if word in adversative_list:
                return True
        return False

    def extract_clause_opinion(self, std_p: float):
        text_opinions_list = []
        for data in self.data_list:
            opinion_list_of_text = []
            main_senti = data['sentiment']
            clause_words_info_list = self._get_clause_with_constituency(data)
            is_pure_opinion, main_shap_standard, main_senti_opinion_list_of_text = self.extract_clause_opinion_for_text_given_senti(senti=main_senti, clause_words_info_list=clause_words_info_list, data=data, std_p=std_p)
            opinion_list_of_text.extend(main_senti_opinion_list_of_text)
            if not is_consider_multi_sentiment_through_adversative:
                text_opinions_list.append(opinion_list_of_text)
                continue;
            # If there is only one clause, or the main sentiment ends with "default", there is no need to discuss secondary sentiments:
            if not is_pure_opinion:
                is_contain_multi_sentiment = self.may_contain_multi_sentiment(data)
                # If a transition is present, begin exploring secondary sentiments:
                if is_contain_multi_sentiment:
                    # Determine the secondary sentiment
                    if abs(main_senti) == 1:
                        second_senti = -main_senti
                    else:
                        sorted_indices, sorted_probs = self.sort_senti_prob(data['senti_prob'])
                        second_senti = sorted_indices[1] - 1
                    # Extract opinions for the secondary sentiment:
                    _, _, second_senti_opinion_list_of_text = self.extract_clause_opinion_for_text_given_senti(senti=second_senti, clause_words_info_list=clause_words_info_list, data=data, std_p=std_p, preset_standard=main_shap_standard)
                    for opinion in second_senti_opinion_list_of_text:
                        # If no suitable opinion is found for the secondary sentiment, do not add it to the list:
                        if opinion['actual_index'] == [-1, -1] or opinion['shap_value'] < main_shap_standard:
                            continue
                        # Check if there is an existing opinion with the same clause_index
                        existing_opinion = next((op for op in opinion_list_of_text if op['clause_index'] == opinion['clause_index']), None)
                        if existing_opinion is None:
                            # If it does not exist, add it directly
                            opinion_list_of_text.append(opinion)
                        else:
                            # If it exists, compare the SHAP values
                            if opinion['shap_value'] > existing_opinion['shap_value']:
                                opinion_list_of_text.remove(existing_opinion)
                                opinion_list_of_text.append(opinion)

            text_opinions_list.append(opinion_list_of_text)

        return text_opinions_list

    def _load_data(self):
        text_list = []
        senti_list = []
        senti_prob_list = []
        word_pieces_list = []
        neg_shaps_list = []
        neu_shaps_list = []
        pos_shaps_list = []

        # Load the preprocessed NLP analysis
        tagged_data = {}
        if self.tagged_file!="":
            with open(self.tagged_file, 'r') as tagged_file:
                for line in tagged_file:
                    data = json.loads(line.strip())
                    tagged_data[data['text']] = data

        with open(self.shap_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['text']
                text_list.append(text)
                senti = data['senti']
                senti_list.append(senti)
                senti_prob = data['senti_prob']
                senti_prob_list.append(senti_prob)
                word_pieces = [word.strip() for word in data['word_piece']]
                word_pieces_list.append(word_pieces)
                shaps_arr = data['shaps_arr']
                neg_shaps = [item[0] for item in shaps_arr[1:]]
                neu_shaps = [item[1] for item in shaps_arr[1:]]
                pos_shaps = [item[2] for item in shaps_arr[1:]]
                neg_shaps_list.append(neg_shaps)
                neu_shaps_list.append(neu_shaps)
                pos_shaps_list.append(pos_shaps)

        data_list = []
        for text, senti, senti_prob, word_pieces, neg_shaps, neu_shaps, pos_shaps \
                in zip(text_list, senti_list, senti_prob_list, word_pieces_list, neg_shaps_list, neu_shaps_list,pos_shaps_list):

            # Use the preprocessed NLP analysis
            if text in tagged_data:
                tagged_entry = tagged_data[text]
                pred_text = tagged_entry['pred_text']
                parse = nlp.annotate(pred_text, properties={
                    'annotators': 'tokenize, ssplit, pos, lemma, parse, coref',
                    'outputFormat': 'json',
                    'timeout': 30000,
                    'tokenize.whitespace': 'true'  # Use whitespace tokenization
                })
                sentences = json.loads(parse)['sentences']
                iter_word = pred_text.split()
                iter_pos = [pos_info[2] for pos_info in tagged_entry['POS']]  # Use the pre-labeled POS

                # Recalculate characterOffsetBegin and characterOffsetEnd in the original text
                iter_character_offset_begin = []
                iter_character_offset_end = []
                current_offset = 0
                for word in iter_word:
                    start_index = text.find(word, current_offset)
                    if start_index == -1:
                        raise ValueError(f"Unable to find the word '{word}' in the original text '{text}'.")
                    end_index = start_index + len(word)
                    iter_character_offset_begin.append(start_index)
                    iter_character_offset_end.append(end_index)
                    current_offset = end_index

            # NLP analysis for non-preprocessed one
            else:
                parse = nlp.annotate(text, properties={
                    'annotators': 'tokenize, ssplit, pos, lemma, parse, coref',
                    'outputFormat': 'json',
                    'timeout': 30000
                })
                sentences = json.loads(parse)['sentences']
                iter_word = list(map(lambda x: x['word'], chain(*map(lambda x: x['tokens'], sentences))))
                iter_pos = list(map(lambda x: x['pos'], chain(*map(lambda x: x['tokens'], sentences))))
                iter_character_offset_begin = list(map(lambda x: x['characterOffsetBegin'], chain(*map(lambda x: x['tokens'], sentences))))
                iter_character_offset_end = list(map(lambda x: x['characterOffsetEnd'], chain(*map(lambda x: x['tokens'], sentences))))


            # SHAP values for each sentiment
            iter_neg_shap = list(zip(word_pieces, neg_shaps))
            iter_neu_shap = list(zip(word_pieces, neu_shaps))
            iter_pos_shap = list(zip(word_pieces, pos_shaps))
            # Generate the index for each word
            iter_index = list(map(lambda x: x['index'], chain(*map(lambda x: x['tokens'], sentences))))
            parse_list = [sentence["parse"] for sentence in sentences]
            parse_string = " ".join(parse_list)
            words_info = []
            start_index = 0
            for word, pos, index, characterOffsetBegin, characterOffsetEnd in zip(iter_word, iter_pos, iter_index,
                                                                                  iter_character_offset_begin,
                                                                                  iter_character_offset_end):
                neg_shap = sum(shap[1] for shap in iter_neg_shap[start_index:start_index + len(word)])
                neu_shap = sum(shap[1] for shap in iter_neu_shap[start_index:start_index + len(word)])
                pos_shap = sum(shap[1] for shap in iter_pos_shap[start_index:start_index + len(word)])
                word_info = [index - 1, word, pos, neg_shap, neu_shap, pos_shap, [characterOffsetBegin, characterOffsetEnd]]
                words_info.append(word_info)
                start_index = start_index + len(word)

            data_list.append({'sentence': text,
                              'sentiment': senti,
                              'senti_prob': senti_prob,
                              'constituency': parse_string,
                              'words_info': words_info})

        return data_list


# Write after analyzing all the texts:
def analysis_shap_1(model: Model, text_list: list, sentiment_list: list):
    def func(x):
        data = DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    piece_list, shap_list = [], []
    explainer = Explainer(func, Text(DataGenerator.tokenizer))
    for text, sentiment in zip(text_list, sentiment_list):
        shap_value = explainer([text])
        piece, shap = shap_value.data[0][1:-1], shap_value.values[0, :, sentiment+1][1:-1]
        print(piece, shap)
        shap_list.append(shap)
        piece_list.append(piece)
    return list(map(lambda x: list(zip(x[0], x[1])), zip(piece_list, shap_list)))

# While analyzing SHAP, intermediate results are recorded to prevent the program from crashing due to resource issues, ensuring that early results are not lost:
def analysis_shap_2(model: Model, text_list: list, sentiment_list: list, shap_file: str):
    # Load parsed files：
    parsed_text_list = []
    if os.path.exists(shap_file):
        with open(shap_file, 'r') as file:
            for line in file:
                line = line.strip()
                parsed_text_shaps = json.loads(line)
                parsed_text_list.append( parsed_text_shaps['text'] )

    pred_outputs = model.predict(DataGenerator(text_list, [0 for _ in text_list]))
    pred_senti = list( map(lambda x: int(x) - 1, np.where(pred_outputs == np.max(pred_outputs, axis=1).reshape(-1, 1))[1]))

    # Start parsing：
    def func(x):
        data = DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    explainer = Explainer(func, Text(DataGenerator.tokenizer))
    size = len(text_list)
    for i in range(size):
        text = text_list[i]
        sentiment = sentiment_list[i]
        parsed_text = parsed_text_list[i] if i<len(parsed_text_list) else None
        # Already parsed, skip：
        if text==parsed_text:
            continue
        #Conduct parsing：
        else:
            shap_value = explainer([text])
            piece, shap = shap_value.data[0][1:-1], shap_value.values[0, :, sentiment+1][1:-1]
            piece_shap_list = list( zip(piece,shap) )
            print(piece_shap_list)
            parsed_text =  {}
            parsed_text['text'] = text
            parsed_text['shaps'] = piece_shap_list

            senti_probabilities = pred_outputs[i].flatten()
            parsed_text_2 = {
                'text': text,
                'senti': pred_senti[i],
                'senti_prob': senti_probabilities.tolist(),
                'word_piece': piece.tolist(),
                'shaps_arr': shap_value.values[0].tolist()
            }
            # Write to the file
            with open(shap_file, 'a') as file2:
                file2.write(json.dumps(parsed_text_2) + "\n")

    print(f"SHAP analysis results have been saved to {shap_file} !")


def extract_opinion(target_name, best_model, sentiment_fname, shap_file, opinion_file):
    print(f"Start extract_opinion()!")
    print(f"Model: {best_model}")
    print(f"Sentiment File: {sentiment_fname}")
    print(f"Shap Final File: {shap_file}")

    # Step 2.1 ：SHAP Analysis
    text_sentiment_data = pd.read_csv(sentiment_fname)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    analysis_shap_2(best_model, text_list, sentiment_list, shap_file)

    # Step 2.2 ：Extract representative word based on SHAP values.
    extractor = OpinionExtractor(target_name, shap_file)
    text_list = extractor.get_texts()
    extractor.is_consider_multi_sentiment_through_adversative = True
    opinion_list = extractor.extract_clause_opinion(std_p=std_p_value)
    with open(opinion_file, 'w') as file:
        for i in range(len(opinion_list)):
            text_unit = {"text": text_list[i], "opinions": opinion_list[i]}
            file.write(f"{json.dumps(text_unit,ensure_ascii=False)}\n")
    print(f"Opinion File: {opinion_file}")


if __name__ == "__main__":
    """
        second step work, calculate SHAP value, extract representative words
    """

    target_name = 'phone'
    model_name = f'{target_name}_model'
    include_text_cnn = True
    best_model = EASTER_ch.load_model(model_name, *EASTER_ch.model_params[target_name], include_text_cnn)

    sentiment_fname = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    shap_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps.txt'
    opinion_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion_test.txt'

    extract_opinion(target_name, best_model, sentiment_fname, shap_file, opinion_file)