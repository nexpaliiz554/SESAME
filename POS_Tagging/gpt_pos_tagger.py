import json
import os
import openai
import time

# Fill in your api_key
client = openai.OpenAI(api_key='')

def get_completion(prompt, model="gpt-4o"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def tag_pos(text):
    prompt = f"Segment the following text into words and assign Part-of-Speech tags for each word in the text following the Penn Treebank Guideline.\n" \
               f"Text: {text}\n" \
               f"Output format: [Word] - [POS tag]"
    while True:
        try:
            response = get_completion(prompt)
            # Check if the response has a value
            if response is not None:
                break  # If the response has a value, jump out of the loop
        except Exception as e:
            print("An exception has occurred:", e)
            sec = 45
            print("Rate limit exceeded. Retrying in {} seconds...".format(sec))
            time.sleep(sec)

    return response

def analysis_for_file(input_fname,gpt_output_fname):
    start_time = time.time()
    print("Start analysis_for_file()")

    text_resed_list = []
    if not os.path.exists(gpt_output_fname):
        with open(gpt_output_fname,'w') as file:
            file.write("")
    else:
         with open(gpt_output_fname,'r') as file:
            for line in file:
                as_res = json.loads(line)
                text = as_res['text']
                text_resed_list.append(text)

    text_need_res_list = []
    with open(input_fname,'r') as file:
        for line in file:
            elems = line.split("\t")
            text_need_res_list.append(elems[0].strip())

    for i in range(len(text_need_res_list)):
        text_need_res = text_need_res_list[i]
        text_in_resed = text_resed_list[i] if i < len(text_resed_list) else None
        if text_need_res==text_in_resed:
            continue;

        pos_tagging = tag_pos(text_need_res)
        res = {'text':text_need_res,
               'POS':pos_tagging}
        print(res)

        with open(gpt_output_fname,'a') as file:
            file.write(json.dumps(res, ensure_ascii=False) + "\n")

    end_time = time.time()
    total_time = end_time - start_time
    print()
    print("analysis_for_file() Over！")
    print("Total time consumption:", total_time, "s")




if __name__ == '__main__':

    input_fname = "input/camera_as_test.tsv"
    gpt_output_fname = "output/camera_as_test_gpt_output.txt"
    analysis_for_file(input_fname,gpt_output_fname)

