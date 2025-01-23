import re
import json
from collections import defaultdict
from itertools import chain
from pycorenlp import StanfordCoreNLP as pyCoreNLP
import shutil


# Normalize original json data
def process_pos_data(input_file, target):

    output_file = f'{target}.txt'
    # Remove noise and normalize the POS section for further operations
    def clean_pos_text(pos_text):
        # match format: "[num. word - POS]" 
        pos_pattern = r'(\d+)\.\s*(\S+)\s*-\s*([A-Za-z\s$()]+)'
        matches = re.findall(pos_pattern, pos_text)

        pos_list = []
        for match in matches:
            index = int(match[0])
            word = match[1]
            pos_tag = match[2].strip().split(' ')[0]
            pos_list.append([index, word, pos_tag])
        
        return pos_list

    # Normalize original json data
    def normalize_json_data(input_json):
        text = input_json["text"]
        cleaned_pos = clean_pos_text(input_json["POS"])
        pred_text = ' '.join([word[1] for word in cleaned_pos])

        output_json = {
            "text": text,
            "pred_text": pred_text,
            "POS": cleaned_pos
        }

        return output_json

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            input_json = json.loads(line.strip())
            output_json = normalize_json_data(input_json)
            json.dump(output_json, outfile, ensure_ascii=False)
            outfile.write('\n')



# Mapping tag and identify tag neither in valid_tags and expanded_pos_tags
def process_pos_tags(target):

    input_file=f'{target}.txt'
    output_file_invalid = f'remain_invalid_{target}.txt'
    outfile_mapped = f'output_{target}.txt'

    expanded_list_matches = 0
    expanded_tag_count = defaultdict(int)

    # Chinese corresponding labels
    valid_tags = ['AD', 'AS', 'BA', 'CC', 'CD', 'CS','DEC', 'DEG', 'DER','DEV', 'DT', 'ETC', 'EM',
                'FW', 'IC', 'IJ', 'JJ', 'LB', 'LC','M', 'MSP', 'NN', 'NOI', 'NR', 'NT',
                'OD', 'ON', 'P', 'PN', 'PU', 'SB','SP', 'URL', 'VA', 'VC', 'VE', 'VV']

    # English corresponding labels
    expanded_pos_tags = ["EX", "IN", "JJR", "JJS", "LS", "MD", "NNS", "NNP", "NNPS", "PDT", "POS" ,"PRP",
                        "PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ",
                        "WDT","WP","WP$","WRB"]

    # Data where the judgment is correct but the expression is not standard in 4-o
    tag_mapping = {
        'ADJ': 'JJ', 
        'ADV': 'RB',
        'Adjective':'JJ',
        'Adverb':'RB',
        'Verb':'VB',
        'Noun':'NN',
    }
    # Identify the incorrectly judged data
    def check_pos_tags(data, expanded_list_matches):
        invalid_entries = []

        for i, (idx, word, tag) in enumerate(data["POS"]):
            word = word.strip() 
            if tag == 'DE' and word == '的':
                tag = 'DEG' 
            if tag == 'DE' and word == '得':
                tag = 'DEG'
            if tag in tag_mapping:
                tag = tag_mapping[tag]
            if tag in expanded_pos_tags:
                expanded_list_matches += 1
                expanded_tag_count[tag] += 1
            elif tag not in valid_tags and tag not in expanded_pos_tags:
                invalid_entries.append((word, tag))
            data["POS"][i] = (idx, word, tag)
        
        return data, invalid_entries

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file_invalid, 'w', encoding='utf-8') as outfile_invalid, \
         open(outfile_mapped, 'w', encoding='utf-8') as outfile_mapped:
        
        for line in infile:
            data = json.loads(line.strip())
            data, invalid_tags = check_pos_tags(data, expanded_list_matches)
            
            if invalid_tags:
                print(f"Unmatched tags: {invalid_tags}")

            json.dump(data, outfile_mapped, ensure_ascii=False)
            outfile_mapped.write('\n')

            result = {
                "text": data['text'],
                "pred_text": data['pred_text'],
                "POS": data["POS"]
            }
            outfile_invalid.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Count the number of tags that match expanded_pos_tags
    print(f"Found {expanded_list_matches} tags in expanded POS list.")
    for tag, count in expanded_tag_count.items():
        print(f"Tag '{tag}' matched {count} times.")


# Identity the split not match text and replace them with coreNLP tagged
def identify_unmatched_texts(target):

    # Start the CoreNLP server
    nlp = pyCoreNLP('http://localhost:9876')

    input_file = f'output_{target}.txt' 
    output_file = f'unmatched_{target}.txt'
    temp_file = f'temp_{target}.txt'

    # The split and text did not fully match
    def compare_text_with_split(text, split):
        reconstructed_text = ''.join(split.split())
        return reconstructed_text == text

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile, open(temp_file, 'w', encoding='utf-8') as tempfile:
        for line in infile:
            data = json.loads(line.strip())
            text = data['text']
            split = data['split']
            
            if not compare_text_with_split(text, split):
                parse = nlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma, parse, coref',
            'outputFormat': 'json',
            'timeout': 30000})
        
                sentences = json.loads(parse)['sentences']
                iter_word = list(map(lambda x: x['word'], chain(*map(lambda x: x['tokens'], sentences))))
                iter_pos = list(map(lambda x: x['pos'], chain(*map(lambda x: x['tokens'], sentences))))

                # Construct POS data format
                pos_list = [[i + 1, iter_word[i], iter_pos[i]] for i in range(len(iter_word))]

                parse_list = [sentence["parse"] for sentence in sentences]
                parse_string = " ".join(parse_list)
                
                data['split'] = parse_string
                data['POS'] = pos_list

                result = {
                    'text': data['text'],
                    'pred_text': parse_string,
                    'POS': pos_list
                }

                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            tempfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    shutil.move(temp_file, input_file)

# convert the original file into a more readable version
def format_origin_data(input_file,target):

    process_pos_data(input_file, target)
    process_pos_tags(target)
    identify_unmatched_texts(target)


# Use CoreNLP to annotate part-of-speech tags
def perform_CoreNLP_tagging(target):

    # Start the CoreNLP server
    nlp = pyCoreNLP('http://localhost:9876')

    input_file = f'output_{target}.txt'
    output_file = f'nlp_{target}.txt'

    # Use CoreNLP to perform part-of-speech tagging on the sentences
    text_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text_list.append(data)

    result_list = []       
    for data in text_list:
        text = data['pred_text']
        parse = nlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma, parse, coref',
            'outputFormat': 'json',
            'timeout': 30000,
            'tokenize.whitespace': 'true'
        })
        
        sentences = json.loads(parse)['sentences']
        iter_word = list(map(lambda x: x['word'], chain(*map(lambda x: x['tokens'], sentences))))
        iter_pos = list(map(lambda x: x['pos'], chain(*map(lambda x: x['tokens'], sentences))))

        # Construct POS data format
        pos_list = [[i + 1, iter_word[i], iter_pos[i]] for i in range(len(iter_word))]
        
        result = {
            'text': data['text'],
            'pred_text': text,
            'POS': pos_list
        }
        
        result_list.append(result)

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in result_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')  # Ensure Chinese characters are not encoded

    print(f"Results saved to {output_file}")


# Process the 4o data to align with CoreNLP
def process_pos_tags_with_nlp(target):

    # Change the corresponding English markers in 4o to their Chinese equivalents
    pos_mapping = {
        "NNP": "NR", "NNPS": "NR",
        "NNS": "NN",
        "RB": "AD", "RBS": "AD", "WP": "AD", "RBR": "AD", "RBS": "AD",
        "VB": "VV", "MD": "VV", "VBP": "VV", "VBN": "VV", "VBG": "VV", "VBZ": "VV",
        "JJR": "JJ", "JJS": "JJ",
        "PRP": "PN", "PRP$": "PN",
        "UH": "SP"
    }

    outfile_mapped = f'output_{target}.txt'
    input_file_nlp = f'nlp_{target}.txt'
    output_file = f'processed_output_{target}.txt'

    with open(input_file_nlp, 'r', encoding='utf-8') as infile_nlp:
        nlp_data = [json.loads(line.strip()) for line in infile_nlp]

    with open(outfile_mapped, 'r', encoding='utf-8') as infile_output, open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(infile_output):
            data_output = json.loads(line.strip())  
            data_nlp = nlp_data[idx] 

            pos_list_output = data_output["POS"]
            pos_list_nlp = data_nlp["POS"]
            
            for i, pos in enumerate(pos_list_output):
                word = pos[1]  
                original_tag = pos[2]  

                if original_tag in pos_mapping:
                    pos[2] = pos_mapping[original_tag] 
                
                if word in ["是", "就是"]:
                    pos[2] = pos_list_nlp[i][2] 
                
                elif word in ["的", "得", "地"]:
                    pos[2] = pos_list_nlp[i][2] 

                elif pos[2] == "JJ":
                    pos[2] = pos_list_nlp[i][2] 

            json.dump(data_output, outfile, ensure_ascii=False)
            outfile.write('\n') 


# using 4o's result to replace coreNLP's obvious error
def process_files(target, output_result_file):

    noun_pos_tags = {'NN', 'NR', 'NT'}
    verb_pos_tags = {"VC", "VE", "VV"}

    output_file = f'processed_output_{target}.txt'
    nlp_file = f'nlp_{target}.txt'

    with open(output_file, 'r', encoding='utf-8') as f1, open(nlp_file, 'r', encoding='utf-8') as f2:
        output_lines = f1.readlines()
        nlp_lines = f2.readlines()

    if len(output_lines) != len(nlp_lines):
        print("The line counts of the two files are inconsistent; please check the files")
        return

    matched_results = []
    
    for line_output, line_nlp in zip(output_lines, nlp_lines):
        data_output = json.loads(line_output.strip())
        data_nlp = json.loads(line_nlp.strip())

        pos_output = data_output['POS']
        pos_nlp = data_nlp['POS']

        if len(pos_output) != len(pos_nlp):
            print(f"Skip the line where the length of the POS tag list is inconsistent: {data_output['text']}")
            continue

        matched_pos = []
        for i, (pos_o, pos_n) in enumerate(zip(pos_output, pos_nlp)):
            if pos_o[2] in noun_pos_tags and pos_n[2] in verb_pos_tags:
                matched_pos.append((i + 1, pos_o[1], pos_o[2], pos_n[2]))

        if matched_pos:
            result = {
                'text': data_output['text'],
                'pred_text': data_output['pred_text'],
                'POS': pos_output
            }
            matched_results.append(result)

    with open(output_result_file, 'w', encoding='utf-8') as out_f:
        for result in matched_results:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"The matching results have been saved to {output_result_file}")

# generate CoreNLP's result and use 4o's result to replace coreNLP's obvious error
def finding_error(target, output_result_file):
    perform_CoreNLP_tagging(target)
    process_pos_tags_with_nlp(target)
    process_files(target,output_result_file)

if __name__ == "__main__":

    """
    step1 : convert the original file into a more readable version
    step2 : using 4o's result to replace coreNLP's obvious error
    """

    target1="phone"
    target2="camera"

    # when you want change file ,you only need to change this variable{target1,target2}
    target = target1

    # input file
    input_file = f'{target}_as_test_gpt_output.txt'
    #output file
    output_result_file=f'{target}_tagged.txt'

    format_origin_data(input_file,target)
    finding_error(target,output_result_file)
