import json
import os
from collections import defaultdict
from itertools import chain
from config import *
from extract_opinion_en import OpinionExtractor as EnglishOpinionExtractor
from extract_opinion_en import nlp as nlp_enlgish
from extract_opinion_en import OpinionExtractor as ChineseOpinionExtractor
from extract_opinion_en import nlp as nlp_chinese

# Configure java environment
os.environ['PATH'] = f'{os.environ["PATH"]}:{JAVA_PATH}/bin:{JAVA_PATH}/jre/bin'

def extract_aspect(target_name:str, lang:str, text_list:list,  opinion_list: list, potential_aspect_boolean_list: list = []):
    input_fname = f'{BASE_DIR}/data/pred_aspect/temp/{target_name}_opinion.txt'
    output_fname = f'{BASE_DIR}/data/pred_aspect/temp/{target_name}_aspect.txt'
    en_dictionary_fname = f'{BASE_DIR}/data/pred_aspect/dictionary_en/'
    ch_dictionary_fname = f'{BASE_DIR}/data/pred_aspect/dictionary_ch/'
    dictionary_map = {"ch": ch_dictionary_fname, "en": en_dictionary_fname}
    phone_pos_optimization = f'{BASE_DIR}/data/gpt_POS_supplement/phone_tagged.txt'
    camera_pos_optimization = f'{BASE_DIR}/data/gpt_POS_supplement/camera_tagged.txt'
    pos_optimization_map = {"phone": phone_pos_optimization,
                            "camera": camera_pos_optimization,
                            "laptop":"",
                            "rest16":"",}
    jar_fname = f'{BASE_DIR}/data/pred_aspect/SentiAspectExtractor.jar'
    if len(potential_aspect_boolean_list)==len(opinion_list):
        with open(input_fname, 'w', encoding='utf-8') as f:
            for text, opinions, potential_aspect_booleans in zip(text_list, opinion_list,potential_aspect_boolean_list):
                f.write(text)
                f.write('\t')
                opinion_string_list = [
                    f'{",".join(map(str, opinion))}(SPA)' if boolean
                    else f'{",".join(map(str, opinion))}'
                    for opinion, boolean in zip(opinions, potential_aspect_booleans)
                ]
                opinion_string = '; '.join(opinion_string_list)
                f.write(opinion_string)
                f.write('\n')
    else:
        with open(input_fname, 'w', encoding='utf-8') as f:
            for text, opinions in zip(text_list, opinion_list):
                f.write(text)
                f.write('\t')
                f.write('; '.join(map(lambda x: ','.join(map(str, x)), opinions)))
                f.write('\n')

    jvm_params = '-Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -XX:+UseStringDeduplication'
    params = {
        '-jar': jar_fname,
        '-inputfile': input_fname,
        '-outputfile': output_fname,
        '-lang': lang,
        '-dict': dictionary_map[lang],
        '-setloadpretag': pos_optimization_map[target_name],
    }
    command = f'java {jvm_params} ' + ' '.join(f'{key} {value}' for key, value in params.items() if value)
    os.system (command)
    with open(output_fname, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_opinion_sentiment_map(text_opinions):
    opinion_sentiment_map = defaultdict(list)
    for opinion in text_opinions:
        opinion_sentiment_map[tuple(opinion["actual_index"])].append((opinion["sentiment"],opinion["shap_value"]))
    return dict(opinion_sentiment_map)


oe_en = EnglishOpinionExtractor()
oe_ch = ChineseOpinionExtractor()
threshold = 0.2
def is_sentimantal_aspect(lang: str, arr: [], text: str):
    if lang == 'en':
        oe = oe_en
        nlp = nlp_enlgish
        annotations = json.loads(nlp.annotate(text, properties={'annotators': 'lemma', 'outputFormat': 'json', 'tokenize.whitespace': 'true'}))
        lemma_tokens = [token['lemma'] for s in annotations['sentences'] for token in s['tokens']]
        aspect_lemma = "_".join(lemma_tokens[arr[0]:arr[1]])
        priori_sentiment = oe._get_prime_priori_value_with_pos(word=aspect_lemma, pos="NN")
        return priori_sentiment > threshold
    elif lang == 'ch':
        oe = oe_ch
        nlp = nlp_chinese
        parse = nlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json',
            'timeout': 30000,
        })
        sentences = json.loads(parse)['sentences']
        iter_word = list(map(lambda x: x['word'], chain(*map(lambda x: x['tokens'], sentences))))
        aspect_lemma = "".join(iter_word[arr[0]:arr[1]])
        return oe.sentiwordnet.__contains__(aspect_lemma)


# Organize and the aspect-sentiment pairs
def build_as_list(lang, senti_prob, opinion_sentiment_map, aspects_of_text,text: str=None):

    def has_overlap(aspect1, aspect2):
        return not (aspect1[1] <= aspect2[0] or aspect2[1] <= aspect1[0])

    def merge_aspects(aspect1, aspect2):
        return (max(aspect1[0], aspect2[0]), min(aspect1[1], aspect2[1]))

    def get_no_intersecting_aspects(aspect_set):
        merged_aspects = []
        for aspect in aspect_set:
            merged = False
            for i in range(len(merged_aspects)):
                if has_overlap(aspect, merged_aspects[i]):
                    merged_aspects[i] = merge_aspects(aspect, merged_aspects[i])
                    merged = True
                    break
            if not merged:
                merged_aspects.append(aspect)
        return merged_aspects

    # Store aspects with different sentiments
    aspect_sets = [set(), set(), set()]

    sentiment_order = sorted([(-1, senti_prob[0]), (0, senti_prob[1]), (1, senti_prob[2])], key=lambda x: -x[1])
    main_sentiment = sentiment_order[0][0]

    # Find the opinion with the largest SHAP value
    max_shap = float('-inf')
    max_shap_opinion = None
    for aspect in aspects_of_text:
        opinion_tuple = tuple(aspect["opinion"])
        if opinion_tuple in opinion_sentiment_map:
            sentiment_shap_tuples_of_opinion = opinion_sentiment_map[opinion_tuple]
            for sentiment_shap_tuple in sentiment_shap_tuples_of_opinion:
                shap = sentiment_shap_tuple[1]
                if shap > max_shap:
                    max_shap = shap
                    max_shap_opinion = opinion_tuple

    default_aspect_set = set() # Record the aspects derived from frequent itemsets
    for aspect in aspects_of_text:
        aspect_tuple = tuple(aspect["aspect"])
        opinion_tuple = tuple(aspect["opinion"])
        if opinion_tuple in opinion_sentiment_map:
            sentiment_shap_tuples_of_opinion = opinion_sentiment_map[opinion_tuple]
            for sentiment_shap_tuple in sentiment_shap_tuples_of_opinion:
                sentiment = sentiment_shap_tuple[0]
                shap = sentiment_shap_tuple[1]
                # If the aspect is (-1, -1), retain it only if it comes from max_shap_opinion
                if aspect_tuple == (-1, -1):
                    if opinion_tuple == max_shap_opinion:
                        aspect_sets[sentiment + 1].add(aspect_tuple)
                else:
                    aspect_sets[sentiment + 1].add(aspect_tuple)
        else:
            default_aspect_set.add(aspect_tuple)

    # Add aspects from aspect_set in the order of sentiments
    as_list = []
    for sentiment, _ in sentiment_order:
        aspect_set = aspect_sets[sentiment + 1]
        aspect_set = get_no_intersecting_aspects(aspect_set)
        for aspect in aspect_set:
            as_list.append({'sentiment': sentiment, 'aspect': list(aspect)})

    # When syntactic parsing fails to identify any explicit aspect, use the default aspect from the text as a fallback
    if len(as_list)==1 and as_list[0]['aspect']==[-1,-1] and len(default_aspect_set)!=0:
        as_list = []
        for aspect in default_aspect_set:
            as_list.append({'sentiment': main_sentiment, 'aspect': list(aspect)})

    # Sentimental nouns cannot be used as aspects
    as_list_filtered = [as_pair for as_pair in as_list if not is_sentimantal_aspect(lang, as_pair['aspect'], text)]
    as_list = as_list_filtered

    # Use implicit aspect as a fallback
    if len(as_list)==0:
        as_list.append({'sentiment': main_sentiment, 'aspect': [-1,-1]})

    return as_list



def extract_as_pair(target_name,lang,shap_file,opinion_fpath,res_fpath):
    print(f"Start extract_as_pair()!")
    print(f"Target Name: {target_name}")
    print(f"Target Language: {lang}")
    print(f"Opinion File: {opinion_fpath}")

    # Read sentiment prediction probability.
    senti_prob_list = []
    with open(shap_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            senti_prob_list.append(data['senti_prob'])

    # Read opinions.
    text_list = []
    opinion_sentiment_map_list = []
    opinion_index_list = []
    opinion_potential_aspect_list = []
    with open(opinion_fpath, 'r') as file:
        for line in file:
            data_unit = json.loads( line.strip() )
            text = data_unit["text"]
            text_list.append(text)
            text_opinions = data_unit["opinions"]
            opinion_sentiment_map_list.append( get_opinion_sentiment_map(text_opinions) )
            opinion_index_list.append( [opinion["actual_index"] for opinion in text_opinions] )
            opinion_potential_aspect_list.append( [opinion["is_potential_aspect"] for opinion in text_opinions] )

    # Extract aspect for opinion.
    aspects_list = extract_aspect(target_name, lang, text_list, opinion_index_list, opinion_potential_aspect_list)

    # Organize (a, s) pairs a:aspect; s:sentiment;
    res_list = []
    for text, senti_prob, opinion_sentiment_map, aspects_of_text in zip(text_list, senti_prob_list, opinion_sentiment_map_list, aspects_list):
        as_list = build_as_list(lang, senti_prob, opinion_sentiment_map, aspects_of_text, text)
        res = {'sentence': text,'as_list': as_list}
        res_list.append(res)

    # Write the results to a file.
    with open(res_fpath, 'w') as f:
        for res in res_list:
            f.write(json.dumps(res,ensure_ascii=False)+"\n")
    print(f"Aspect-sentiment pairs output at：{res_fpath}!")


lang_map = {'laptop': 'en',
            'rest16': 'en',
            'phone': 'ch',
            'camera': 'ch' }

if __name__ == "__main__":
    """
        third step work，extract aspect
    """

    target_name = 'laptop'
    lang = lang_map[target_name]
    shap_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps.txt'
    opinion_fpath = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion_test.txt'
    res_fpath = f'{BASE_DIR}/data/pred_aspect/{target_name}_as_test.txt'

    extract_as_pair(target_name,lang,shap_file,opinion_fpath,res_fpath)



