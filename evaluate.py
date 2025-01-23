import json
import copy
from config import *
import csv


class Evaluater:
    def __init__(self, human_label_fname: str):
        self.human_label_fname = human_label_fname
        self.EVALUATE_TYPE = -1
        self.EVALUATE_ACOS = 0
        self.EVALUATE_AS = 1
        if "quad" in human_label_fname:
            self.EVALUATE_TYPE = self.EVALUATE_ACOS
        elif "as" in human_label_fname:
            self.EVALUATE_TYPE = self.EVALUATE_AS
        self.gold_subscripts_list = []
        self.gold_text_list = []
        self._load_data()


    def get_aos_list(self, pred_aos_fname: str):
        aos_list = []
        with open(pred_aos_fname,"r") as f:
            for line in f:
                sentence_res = json.loads( line )
                aos_list.append( sentence_res )
        return aos_list

    def _get_slice(self,indice,text):
        text = text.strip()
        # Split English by spaces.
        if self.EVALUATE_TYPE == self.EVALUATE_ACOS:
            words = text.split(" ")
        # Chinese segmented by character.
        elif self.EVALUATE_TYPE == self.EVALUATE_AS:
            if text.find(" ")!=-1:
                words = text.split(" ")
            else:
                words = list(text)
        if indice[0]==indice[1]==-1:
            slice = ["ImplicitAspect"]
        else:
            slice = [ words[i].lower() for i in range(indice[0],indice[1])]

        return slice

    def _textify_indexing(self,subscripts_list: list):
        textuality_list = copy.deepcopy(subscripts_list)
        for text_unit in textuality_list:
            text = text_unit['sentence']
            unit_list = text_unit['as_list']
            for unit in unit_list:
                unit['aspect'] = self._get_slice(unit['aspect'], text) if 'aspect' in unit else []
                unit['opinion'] = self._get_slice(unit['opinion'], text) if 'opinion' in unit else []
        return textuality_list

    def _is_index_true(self, range1: list, range2: list, standard: str):
        range1 = [range1[0], range1[1] - 1] if range1[0] != -1 else range1
        range2 = [range2[0], range2[1] - 1] if range2[0] != -1 else range2
        if standard == 'strict':
            return range1[0] == range2[0] and range1[1] == range2[1]
        elif standard == 'loose':
            return range1[0] <= range2[0] <= range1[1] or range2[0] <= range1[0] <= range1[1] <= range2[1] or range1[0] <= range2[1] <= range1[1]
        raise RuntimeError('illegal argument')

    def _is_text_true(self, range1: list, range2: list, standard: str):
        if standard == 'strict':
            return range1 == range2
        elif standard == 'loose':
            set1 = set(range1)
            set2 = set(range2)
            return set1.intersection(set2)
        raise RuntimeError('illegal argument')

    def _quota(self, pred_list: list, gold_list: list,aos_map, data_map, is_true):
        TP = FP = GTP = GFN = 0
        for pred_aos, real_aos in zip(map(lambda x: set(map(aos_map, x['as_list'])), pred_list),
                                      map(lambda x: set(map(data_map, x['as_list'])), gold_list)):
            pred_result = list(map(lambda x: len(list(filter(lambda y: is_true(x, y), real_aos))) > 0, pred_aos))
            real_result = list(map(lambda x: len(list(filter(lambda y: is_true(x, y), pred_aos))) > 0, real_aos))
            TP += sum(pred_result)
            FP += len(pred_result) - sum(pred_result)
            GTP += sum(real_result)
            GFN += len(real_result) - sum(real_result)
        P, R = TP / (TP + FP), GTP / (GTP + GFN)
        F = (2 * P * R) / (P + R)
        return P, R, F

    def quota_aos_index(self, aos_list: list, standard: str):
        return self._quota(aos_list,
                           self.gold_subscripts_list,
                           lambda x: (tuple(x['aspect']), tuple(x['opinion']), x['sentiment']),
                           lambda x: (tuple(x['aspect']), tuple(x['opinion']), x['sentiment']),
                           lambda x, y: self._is_index_true(x[0], y[0], standard) and self._is_index_true(x[1], y[1], standard) and x[2] == y[2])


    def quota_aos_text(self, aos_list: list, is_need_textify: bool, standard: str):
        pred_list_texted = self._textify_indexing(aos_list) if is_need_textify else aos_list
        return self._quota(pred_list_texted,
                           self.gold_text_list,
                           lambda x: (tuple(x['aspect']), tuple(x['opinion']), x['sentiment']),
                           lambda x: (tuple(x['aspect']), tuple(x['opinion']), x['sentiment']),
                           lambda x, y: self._is_text_true(x[0], y[0], standard) and self._is_text_true(x[1], y[1],standard) and x[2] == y[2])


    def quota_as_index(self, aos_list: list, standard: str):
        return self._quota(aos_list,
                           self.gold_subscripts_list,
                           lambda x: (tuple(x['aspect']), x['sentiment']),
                           lambda x: (tuple(x['aspect']), x['sentiment']),
                           lambda x, y: self._is_index_true(x[0], y[0], standard) and x[1] == y[1])

    def quota_as_text(self, aos_list: list, is_need_textify: bool, standard: str):
        pred_list_texted = self._textify_indexing(aos_list) if is_need_textify else aos_list
        return self._quota(pred_list_texted,
                           self.gold_text_list,
                           lambda x: (tuple(x['aspect']), x['sentiment']),
                           lambda x: (tuple(x['aspect']), x['sentiment']),
                           lambda x, y: self._is_text_true(x[0], y[0], standard) and x[1] == y[1])

    def quota_a_index(self, aos_list: list, standard: str):
        return self._quota(aos_list,
                           self.gold_subscripts_list,
                           lambda x: (tuple(x['aspect']),),
                           lambda x: (tuple(x['aspect']),),
                           lambda x, y: self._is_index_true(x[0], y[0], standard))

    def quota_a_text(self, aos_list: list, is_need_textify: bool, standard: str):
        pred_list_texted = self._textify_indexing(aos_list) if is_need_textify else aos_list
        return self._quota(pred_list_texted,
                           self.gold_text_list,
                           lambda x: (tuple(x['aspect']),),
                           lambda x: (tuple(x['aspect']),),
                           lambda x, y: self._is_text_true(x[0], y[0], standard))

    def quota_o_index(self, aos_list: list, standard: str):
        return self._quota(aos_list,
                           self.gold_subscripts_list,
                           lambda x: (tuple(x['opinion']),),
                           lambda x: (tuple(x['opinion']),),
                           lambda x, y: self._is_index_true(x[0], y[0], standard))

    def quota_o_text(self, aos_list: list, is_need_textify: bool, standard: str):
        pred_list_texted = self._textify_indexing(aos_list) if is_need_textify else aos_list
        return self._quota(pred_list_texted,
                           self.gold_text_list,
                           lambda x: (tuple(x['opinion']),),
                           lambda x: (tuple(x['opinion']),),
                           lambda x, y: self._is_text_true(x[0], y[0], standard))

    def _parse_index(self,subscript_str):
        return list(map(int, subscript_str.split(',')))

    def _parse_acso(self,entry_string):
        if entry_string == '':
            return None
        entry = {}
        elems = entry_string.split(' ')
        entry["aspect"] = self._parse_index(elems[0])
        entry["category"] = elems[1]
        entry["sentiment"] = int(elems[2])-1
        entry["opinion"] = self._parse_index(elems[3])
        return entry

    def _parse_as(self,entry_string):
        entry = {}
        elems = entry_string.split(' ')
        entry["aspect"] = self._parse_index(elems[0])
        entry["sentiment"] = int(elems[1])
        return entry

    def _parse_line(self,line):
        text = line[0]
        entries = line[1:]
        line_res = {}
        line_res['sentence'] = text
        if self.EVALUATE_TYPE == self.EVALUATE_ACOS:
            parsed_entries = list(map(self._parse_acso, entries))
            parsed_entries = [entry for entry in parsed_entries if entry is not None]
            line_res['as_list'] = parsed_entries
        elif self.EVALUATE_TYPE == self.EVALUATE_AS:
            parsed_entries = list(map(self._parse_as, entries))
            line_res['as_list'] = parsed_entries
        return line_res

    def _load_data(self):
        with open(self.human_label_fname, 'r', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            lines = list(tsv_reader)
        gold_res_list =  list(map(self._parse_line, lines))
        self.gold_subscripts_list = gold_res_list
        self.gold_text_list = self._textify_indexing(gold_res_list)



def evaluate():
    target_name = 'laptop'
    tag_map = {'laptop': 'quad',
               'rest16': 'quad',
               'phone': 'as',
               'camera': 'as' }
    human_label_fname = f'{BASE_DIR}/data/acos/{target_name}_{tag_map[target_name]}_test.tsv'
    pred_aos_fname = f'{BASE_DIR}/data/pred_aspect/{target_name}_as_test.txt'

    extractor = Evaluater(human_label_fname)
    aos_list = extractor.get_aos_list(pred_aos_fname)
    is_need_textify = True;

    # Evaluate the ASPE task with strict match standard.
    print("AS Strict:")
    result = extractor.quota_as_text(aos_list, is_need_textify, 'strict')
    print(f'{result[0]}\t{result[1]}\t{result[2]}')

    # Evaluate the ASPE task with loose match standard.
    # print("AS Loose:")
    # result = extractor.quota_as_text(aos_list, is_need_textify, 'loose')
    # print(f'{result[0]}\t{result[1]}\t{result[2]}')

    # Evaluate the aspect extraction task with strict match standard.
    # print("A Strict:")
    # result = extractor.quota_a_text(aos_list, is_need_textify, 'strict')
    # print(f'{result[0]}\t{result[1]}\t{result[2]}')

    # Evaluate the aspect extraction task with loose match standard.
    # print("A Loose:")
    # result = extractor.quota_a_text(aos_list, is_need_textify, 'loose')
    # print(f'{result[0]}\t{result[1]}\t{result[2]}')



if __name__ == "__main__":
    evaluate()
