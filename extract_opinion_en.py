import json
import os
from itertools import chain
from typing import Iterable

import nltk
import numpy as np
import pandas as pd
from keras.api._v2.keras import Model
from shap import Explainer
from shap.maskers import Text
from stanfordcorenlp import StanfordCoreNLP

import EASTER_en
from EASTER_en import DataGenerator
from EASTER_en import load_model
from config import *


std_p_value = 0.8
is_consider_multi_sentiment_through_adversative = True

nlp = StanfordCoreNLP(STANFORD_CORE_NLP_PATH)
class OpinionExtractor:
    def __init__(self, shap_file: str=None):
        self.sentiment_dictionary = f'{BASE_DIR}/data/dict/SentiWordNet_3.0.0.txt'
        self.shap_file = shap_file
        self.pos_map = {
            "NNS": "n", "NNPS": "n", "NN": "n", "NNP": "n",
            "VBN": "v", "VB": "v", "VBD": "v", "VBZ": "v","VBP": "v", "VBG": "v",
            "JJR": "a", "JJS": "a", "JJ": "a",
            "RBS": "r", "RB": "r", "RP": "r", "WRB": "r", "RBR": "r",
        }
        self.words_lead_to_clause = [""]
        self.pos_tags_lead_to_clause = ["CC", ","]
        self.clause_label_list = ['SBAR','S']
        self.phrase_label_list= ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ',
                                'LST', 'NAC', 'NP', 'NX', 'PP',
                                'PRN', 'PRT', 'QP', 'RRC', 'UCP',
                                'VP', 'WHADJP', 'WHAVP', 'WHNP','WHPP','X']
        # load sentiwordnet
        sentiwordnet = dict()
        with open(self.sentiment_dictionary, 'r', encoding='utf-8') as f:
            for line in map(lambda x: x.split('\t'), f.read().splitlines()[26:-1]):
                for word_number in map(lambda x: x.split('#'), line[4].split(' ')):
                    sentiwordnet.setdefault(word_number[0], {}).setdefault(line[0], []).append([int(word_number[1]), float(line[2]), float(line[3])])
        self.sentiwordnet = {word: {pos: score_list for pos, score_list in info.items()} for word, info in sentiwordnet.items()}
        # laod dataset
        self.data_list = []
        if shap_file!=None:
            self.data_list = self._load_data()

    def get_texts(self):
        """
        Get the input text.
        """
        return [data['sentence'] for data in self.data_list]

    def _get_priori_value(self, word: str):
        """
        Get the priori sentiment value.

        :param word: the word
        :return: return -1 if not exists
        """
        score_list = [score for pos_scores in self.sentiwordnet.get(word, {}).values() for score in pos_scores]
        score = sum(abs(score[1] + score[2]) / score[0] for score in score_list) if score_list else -1
        return score


    def _get_priori_value_with_pos(self, word: str, pos: str):
        """
        Considering the POS, get the priori sentiment value.
        score = Σi=1 to n (PosScore+NegScore)/sense_rank
                n is the number of senses of a word given its POS

        :param word: the word
        :param pos: the pos(part of speech) of the word
        :return: return -1 if not exists
        """
        score_list = self.sentiwordnet.get(word, {}).get(self.pos_map.get(pos, ''), [])
        score = sum(abs(score[1] + score[2]) / score[0] for score in score_list)  if score_list else -1
        return score

    def _get_prime_priori_value_with_pos(self, word: str, pos: str):
        score_list = self.sentiwordnet.get(word, {}).get(self.pos_map.get(pos, ''), [])
        if len(score_list)==0:
            return -1
        else:
            for score in score_list:
                if score[0]==1:
                    return abs(score[1] + score[2])

    def _is_potential_aspect(self, priori_value: float, pos: str):
        """
        To determine if it is a potential aspect,
        the conditions for being a potential aspect are:
            1) There is no prior sentiment value or the prior sentiment value is 0,
            2) It is a noun.
        :param priori_value: the prior sentiment value
        :param pos: the pos of the word
        :return:
        """
        return self.pos_map.__contains__(pos) and self.pos_map[pos]=="n" and priori_value<=0

    def extract_top_k_opinion(self, k: int):
        """
        Extract the top k highest shap value as opinion
        """
        opinion_list = []
        for data in self.data_list:
            senti = data['sentiment']
            one_text_opinion_list = []
            senti_SHAP_index = senti + 4
            words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, data['words_info']), key=lambda x: x[senti_SHAP_index], reverse=True)
            if len(words_info) == 0:
                opinion = {'clause_index': -1,
                           'sentiment': senti,
                           'actual_index': [-1, -1],
                           'shap_value': -1,
                           'is_potential_aspect': False}
                one_text_opinion_list.append( opinion )
            else:
                for x in words_info[0:k]:
                    opinion = {'clause_index': -1,
                               'sentiment': senti,
                               'actual_index': [x[6][0], x[6][-1] + 1],
                               'shap_value': x[senti_SHAP_index],
                               'is_potential_aspect': self._is_potential_aspect(
                                   priori_value=self._get_priori_value(word=x[2]),
                                   pos=x[1])}
                    one_text_opinion_list.append( opinion )

            opinion_list.append( one_text_opinion_list )
        return opinion_list

    def _get_clause_with_comma(self,data):
        '''
        Split text based on commas
        '''
        clauses = []
        current_clause = []
        for item in data['words_info']:
            if item[1] == ",":
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            current_clause.append(item)
        if current_clause:
            clauses.append(current_clause)
        return clauses

    def _get_tree_start_index(self,tree):
        try:
            start_index = int(tree.leaves()[0])
        except Exception:
            start_index = None
        return start_index

    def _get_clause_with_constituency(self, data):
        '''
        Split text based on constituency analysis
        '''
        if len(data['words_info']) == 0:
            return data['words_info']
        constituency_tree_str = data['constituency']
        word_list = [ x[0] for x in data['words_info'] ]
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
        data['constituency'] = constituency_tree_str
        tree = nltk.Tree.fromstring(constituency_tree_str)
        clause_start_index_set = set()
        clause_start_index_canceled_set = set()
        phrase_start_index_set = set()
        for subtree in tree.subtrees():
            start_index = self._get_tree_start_index(subtree)
            if start_index is None:
                continue
            if subtree.label() in self.clause_label_list:
                clause_start_index_set.add(start_index)
            elif subtree.label() in self.phrase_label_list:
                phrase_start_index_set.add(start_index)
            # For SBAR -> S structure, retain only one split:
            if subtree.label()=="SBAR":
                for s_subtree in subtree:
                     if s_subtree.label() == 'S':
                        s_subtree_start_index = self._get_tree_start_index(s_subtree)
                        if s_subtree_start_index!=start_index:
                            clause_start_index_canceled_set.add(s_subtree_start_index)

        clause_start_index_set = clause_start_index_set - clause_start_index_canceled_set
        clauses = []
        current_clause = []
        start_index_set = clause_start_index_set
        if len(clause_start_index_set) == 0:
            start_index_set = phrase_start_index_set
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
            clause_words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, clause_words_info),
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
                       'actual_index': [clause_opinion[6][0], clause_opinion[6][-1] + 1],
                       'shap_value': clause_opinion[senti_SHAP_index],
                       'is_potential_aspect': self._is_potential_aspect(
                           priori_value=self._get_priori_value(word=clause_opinion[2]),
                           pos=clause_opinion[1])}
            opinion_list_for_text.append(opinion)

        if len(opinion_list_for_text) == 0:
            if max_clause_opinion is not None:
                opinion = {'clause_index': -1,
                           'sentiment': senti,
                           'actual_index': [max_clause_opinion[6][0], max_clause_opinion[6][-1] + 1],
                           'shap_value': max_clause_opinion[senti_SHAP_index],
                           'is_potential_aspect': self._is_potential_aspect(
                               priori_value=self._get_priori_value(word=max_clause_opinion[2]),
                               pos=max_clause_opinion[1])}
            else:
                opinion = {'clause_index': -1,
                           'sentiment': senti,
                           'actual_index': [-1, -1],
                           'shap_value': -1,
                           'is_potential_aspect': False}
            is_pure_opinion = True
            opinion_list_for_text.append(opinion)

        is_pure_opinion = is_pure_opinion or len(clause_words_info_list)==1

        return is_pure_opinion, standard, opinion_list_for_text

    def sort_senti_prob(self, senti_prob):
        sorted_indices = sorted(range(len(senti_prob)), key=lambda i: senti_prob[i], reverse=True)
        sorted_probs = [senti_prob[i] for i in sorted_indices]
        return sorted_indices, sorted_probs

    def _is_legal_adversative_for_CC(self,text,tree,adversative_word_index):

        def find_non_punctuation_label(subtree_list, start_idx, direction):
            idx = start_idx + direction
            while 0 <= idx < len(subtree_list):
                if subtree_list[idx].label() not in {',', '.', ';', ':'}:
                    return subtree_list[idx].label()
                idx += direction
            return None

        for subtree in tree.subtrees():
            adversative_tree = None
            adversative_index = -1
            for idx, s_subtree in enumerate(subtree):
                leaf_index = self._get_tree_start_index(s_subtree)
                if adversative_word_index == leaf_index and len(s_subtree) == 1:
                    adversative_tree = subtree
                    adversative_index = idx
            if adversative_tree != None:
                subtree_list = list(adversative_tree)
                left_label = find_non_punctuation_label(subtree_list, adversative_index, -1)
                right_label = find_non_punctuation_label(subtree_list, adversative_index, 1)
                if left_label==right_label:
                    return True
                else:
                    return False

        return False

    def _is_legal_adversative_for_while(self, pos_list, adversative_word_index):
        if pos_list[adversative_word_index].startswith("NN"):
            return False
        if adversative_word_index + 1 < len(pos_list) and pos_list[adversative_word_index + 1] == "VBG":
            return False
        return True

    def may_contain_multi_sentiment(self, data):
        adversative_words = ["but", "however", "though", "although", "while"]
        adversative_indices = []
        lemma_list = [x[2] for x in data['words_info']]
        pos_list = [x[1] for x in data['words_info']]
        constituency_tree_str = data['constituency']
        tree = nltk.Tree.fromstring(constituency_tree_str)

        def find_next_comma(word_list, start_idx):
            return next((i for i in range(start_idx + 1, len(word_list)) if word_list[i] == ","), None)

        for idx, word in enumerate(lemma_list):
            if word not in adversative_words:
                continue;
            if word in ["but", "however", "though", "although"] and idx==0:
                continue;
            if word=="while" and not self._is_legal_adversative_for_while(pos_list,idx):
                continue;
            if pos_list[idx]=="CC":
                _is_legal_adversative = self._is_legal_adversative_for_CC(data['sentence'],tree,idx)
                if _is_legal_adversative:
                    adversative_indices.append(idx)
            else:
                # If there is a comma after the transition, use the first comma as the delimiter; if there is no comma, use the transition word itself as the delimiter:
                comma_index = find_next_comma(lemma_list, idx)
                if comma_index is not None:
                    adversative_indices.append(comma_index)
                else:
                    adversative_indices.append(idx)

        return adversative_indices

    # Helper function to process the opinion_list, select the sentiment with the highest SHAP value, and retain opinion words with consistent sentiment
    def filter_opinions_by_max_shap(self,opinions):
        if opinions:
            # Find the opinion with the highest SHAP value
            max_shap_opinion = max(opinions, key=lambda op: op['shap_value'])
            max_senti = max_shap_opinion['sentiment']
            # Retain representative words with the same sentiment as max_senti
            return [op for op in opinions if op['sentiment'] == max_senti]
        return opinions

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
            # If there is only one clause, or the main sentiment ends with "default", there is no need to discuss secondary sentiments
            if not is_pure_opinion:
                adversative_indices = self.may_contain_multi_sentiment(data)
                # If a transition is present, begin exploring secondary sentiments:
                if adversative_indices:
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
                        if opinion['actual_index']==[-1,-1] or opinion['shap_value']<main_shap_standard:
                            continue
                        # Check if there is an existing opinion with the same clause_index
                        existing_opinion = next( (op for op in opinion_list_of_text if op['clause_index'] == opinion['clause_index']), None)
                        if existing_opinion is None:
                            # If it does not exist, add it directly
                            opinion_list_of_text.append(opinion)
                        else:
                            # If it exists, compare the SHAP values
                            if opinion['shap_value'] > existing_opinion['shap_value']:
                                opinion_list_of_text.remove(existing_opinion)
                                opinion_list_of_text.append(opinion)
                    # Ensure sentiment consistency on one side of the transition
                    last_adversative_index = adversative_indices[-1]
                    left_opinions = [op for op in opinion_list_of_text if op['actual_index'][0] <= last_adversative_index]
                    right_opinions = [op for op in opinion_list_of_text if op['actual_index'][0] > last_adversative_index]
                    left_opinions = self.filter_opinions_by_max_shap(left_opinions)
                    right_opinions = self.filter_opinions_by_max_shap(right_opinions)
                    opinion_list_of_text = left_opinions + right_opinions

            text_opinions_list.append(opinion_list_of_text)

        return text_opinions_list

    def _process_sentence(self, sentence: str) -> str:
        index, words, result = 0, sentence.split(' '), []
        while index < len(words):
            if index < len(words) - 3 and words[index+1] == "'" and words[index+2] in ('t', 'd', 'm', 's', 're', 've', 'll'):
                result.append(''.join(words[index:index+3]))
                index += 3
                continue
            result.append(words[index])
            index += 1
        return ' '.join(result)

    def _index_token(self, sentence: str, token_list: Iterable):
        index, result = 0, []
        for token in token_list:
            token_index = sentence.find(token[0], index)
            if token_index < 0:
                return []
            index = token_index + len(token[0])
            result.append((token[0], token[1], (token_index, index)))
        return result

    def _is_true(self, range1: list, range2: list, which: str):
        range1 = [range1[0], range1[1] - 1] if range1[0] != -1 else range1
        range2 = [range2[0], range2[1] - 1] if range2[0] != -1 else range2
        if which == 'strict':
            return range1[0] == range2[0] and range1[1] == range2[1]
        elif which == 'loose':
            return range1[0] <= range2[0] <= range1[1] or range2[0] <= range1[0] <= range1[1] <= range2[1] or range1[0] <= range2[1] <= range1[1]
        raise RuntimeError('illegal argument')

    def _merge(self, iter_pos: list, iter_lemma: list, iter_neg_shap: list, iter_neu_shap: list, iter_pos_shap: list, iter_index: list):
        if 0 in (len(iter_pos), len(iter_lemma), len(iter_neg_shap), len(iter_neu_shap), len(iter_pos_shap), len(iter_index)):
            return []
        return list(map(lambda x: (
            x[0][0], #0: word
            x[0][1], #1: pos
            x[1], #2: lemma
            sum(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_neg_shap))), #3: neg SHAP value
            sum(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_neu_shap))), #4: neu SHAP value
            sum(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_pos_shap))), #5: pos SHAP value
            list(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_index))) #6: index list
        ), zip(iter_pos, iter_lemma)))

    def _load_data(self):
        text_list = []
        senti_list = []
        senti_prob_list = []
        word_pieces_list = []
        neg_shaps_list = []
        neu_shaps_list = []
        pos_shaps_list = []
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
                in zip(text_list, senti_list, senti_prob_list, word_pieces_list, neg_shaps_list, neu_shaps_list, pos_shaps_list):
            # Process the sentence
            sentence = self._process_sentence(text)
            # Lemmatization
            iter_lemma = list(map(lambda x: x['lemma'],chain(*map(lambda x: x['tokens'], json.loads(nlp.annotate(sentence))['sentences']))))
            # POS tagging
            iter_pos = nlp.pos_tag(sentence)
            # SHAP values for each sentiment
            iter_neg_shap = zip(word_pieces,neg_shaps)
            iter_neu_shap = zip(word_pieces, neu_shaps)
            iter_pos_shap = zip(word_pieces, pos_shaps)
            # Generate the index for each word
            iter_index = map(lambda x: (x[1], x[0]), enumerate(text.split(' ')))
            # Map the indices obtained from CoreNLP parsing
            iter_pos = self._index_token(sentence, iter_pos)
            iter_neg_shap = self._index_token(sentence,iter_neg_shap)
            iter_neu_shap = self._index_token(sentence,iter_neu_shap)
            iter_pos_shap = self._index_token(sentence, iter_pos_shap)
            iter_index = self._index_token(sentence, iter_index)

            # Create a list of dictionaries containing all information
            words_info = self._merge(iter_pos, iter_lemma, iter_neg_shap, iter_neu_shap, iter_pos_shap, iter_index)

            data_list.append({'sentence': text,
                              'processed_sentence': sentence,
                              'sentiment': senti,
                              'senti_prob': senti_prob,
                              'dependency': nlp.dependency_parse(sentence),
                              'constituency': nlp.parse(sentence),
                              'words_info': words_info})
        return data_list

# Write after analyzing all the texts:
def analysis_shap_1(model: Model, text_list: list, sentiment_list: list):
    """
    calculate shapely value of texts using model
    """
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
            parsed_text = {'text': text, 'shaps': piece_shap_list}
            parsed_text_json = json.dumps(parsed_text)

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


def extract_opinion(target_name,best_model,sentiment_fname,shap_file,opinion_file):
    print(f"Start extract_opinion()!")
    print(f"Model: {best_model}")
    print(f"Sentiment File: {sentiment_fname}")
    print(f"Shap Final File: {shap_file}")

    # Step 2.1 ：SHAP Analysis
    text_sentiment_data = pd.read_csv(sentiment_fname)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    analysis_shap_2(best_model,text_list,sentiment_list,shap_file)

    # Step 2.2 ：Extract representative word based on SHAP values.
    extractor = OpinionExtractor(shap_file)
    text_list = extractor.get_texts()
    extractor.is_consider_multi_sentiment_through_adversative = True
    opinion_list = extractor.extract_clause_opinion(std_p=std_p_value)
    with open(opinion_file, 'w') as file:
        for i in range(len(opinion_list)):
            text_unit = {"text": text_list[i], "opinions": opinion_list[i]}
            file.write(f"{json.dumps(text_unit)}\n")

    print(f"Opinion File: {opinion_file}")


if __name__ == "__main__":
    """
        second step work, calculate SHAP value, extract representative words
    """

    target_name = 'laptop'

    model_name = f'{target_name}_model'
    include_text_cnn = True
    best_model = load_model(model_name, *EASTER_en.model_params[target_name], include_text_cnn)

    sentiment_fname = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    shap_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps.txt'
    opinion_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion_test.txt'

    extract_opinion(target_name, best_model,sentiment_fname,shap_file,opinion_file)