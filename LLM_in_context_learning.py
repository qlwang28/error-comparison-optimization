# coding=utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../../")
import csv
import re
import logging
import argparse
import ast

import torch
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)


binary_inference_template = \
"""\
Task Description:
Perform Aspect-Based Sentiment Analysis (ABSA) on customer reviews. ABSA involves identifying specific features or attributes of a product or service (aspects) discussed in review text, and analyzing the sentiment expressed towards these aspects.

Instructions:
1. Identify Aspects: Identify all the distinct features or attributes mentioned in the review. Aspects should be extracted as noun phrase spans.
2. Analyze Sentiment: Assess the sentiment for each identified aspect based on the language used in relation to that aspect. Sentiment should be classified as:
   - Positive (POS): Expresses satisfaction or positive emotions.
   - Negative (NEG): Expresses dissatisfaction or negative emotions.
   - Neutral (NEU): Neither explicitly positive nor negative, or the context does not provide enough information for a clear sentiment.
3. Label the Aspects and Sentiments: Format your findings as a list of lists, where each inner list contains an aspect followed by its sentiment polarity code.

Examples for Guidance:
{}
Review: "{}"
Label: """


triplet_inference_template = \
"""\
Task Description:  
Perform Aspect-Based Sentiment Analysis (ABSA) on customer reviews. ABSA involves identifying specific attributes of a product (aspects) discussed in the review, analyzing the sentiment expressed towards these aspects, and extracting any explicit opinion terms used to describe them.

Instructions:  
1. Identify Aspects: Identify all the distinct features or attributes mentioned in the review. Aspects should be extracted as noun phrase spans. If the aspect is implicit (not explicitly mentioned in the review), output `null` for the aspect.
2. Extract Opinions: Identify the explicit opinion terms associated with each aspect. If the opinion term is implicit (not explicitly mentioned in the review), output `null` for the opinion term.
3. Analyze Sentiment: Assess the sentiment for each identified aspect based on the language used in relation to that aspect. Sentiment should be classified as:  
   - Positive (POS): Expresses satisfaction or positive emotions.  
   - Negative (NEG): Expresses dissatisfaction or negative emotions.  
   - Neutral (NEU): Neither explicitly positive nor negative, or the context does not provide enough information for a clear sentiment.  
4. Label the Aspects, Sentiments, and Opinion Terms: Format your findings as a list of lists, where each inner list contains:
    - Aspect
    - Opinion
    - Sentiment polarity code

Examples for Guidance:
{}

Review: "{}"  
Label: """


quadruple_inference_template = \
"""\
Task Description:
Perform Aspect-Based Sentiment Analysis (ABSA) on customer reviews. ABSA aims to predict all quads (aspect term, aspect category, opinion term, sentiment polarity) for a given review.

Instructions:
1. Identify Aspects: Identify all the distinct features or attributes mentioned in the review. Aspects should be extracted as noun phrase spans. If the aspect is implicit (not explicitly mentioned in the review), output `null` for the aspect.
2. Extract Opinion Terms: Identify the specific words or phrases that convey the sentiment towards each aspect. If the opinion term is implicit (not explicitly mentioned in the review), output `null` for the opinion term.
3. Determine Aspect Categories: Map each identified aspect to a predefined category. Consists of an entity label and attribute label, with possible values like: 
{}
4. Analyze Sentiment: Assess the sentiment for each aspect based on the language used in relation to it. Sentiment should be classified as:
   - positive: Expresses satisfaction or positive emotions.
   - negative: Expresses dissatisfaction or negative emotions.
   - neutral: Neither explicitly positive nor negative, or the context does not provide enough information for a clear sentiment.
5. Label the Quads: Format your findings as a list of quads, where each quad contains:
   - Aspect term
   - Opinion term
   - Aspect category
   - Sentiment polarity

Examples for Guidance:
{}
Review: "{}"
Label: """


def read_csv(file_path, current_len, model_name):
    with open(file_path, "r", encoding='utf-8') as file_obj:
        reader_obj = csv.reader(file_obj)
        lines = [
            row + ([model_name] if i == 0 else [""]) if len(row) == current_len else row for i, row in enumerate(reader_obj)
        ]
    return lines


def write_csv(output_path, lines):
    with open(output_path, "w", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in lines:
            writer.writerow(line)


def in_context_learning_inference(LLM_obeject, sampling_params, prompts):
    responses = LLM_obeject.generate(prompts, sampling_params)
    results = []
    for i in range(len(responses)):
        resp = responses[i].outputs[0].text
        resp = re.sub(r'\[\s+\[', '[[', resp)
        resp = re.sub(r'\]\s+\]', ']]', resp)
        try:
            json_obj = re.search(r'\s*(\[\[.*?\]\])', resp).group(1)
        except Exception as e:
            json_obj = resp.split("\n")[0].strip()
        results.append(str(json_obj))
    return results


def main(args, inference_template, indexs, shot2column):
    model_path = '/home/wangqianlong/models/pre-trained_model/{}'.format(args.MODEL_NAME)
    LLM_obeject = LLM(model=model_path, tensor_parallel_size=1, dtype=torch.bfloat16)
    sampling_params = SamplingParams(max_tokens=512)

    csv_path = os.path.join(args.data_dir, args.domain, "reason_test_bm25_16_shot.csv")
    logger.info("****** TASK : {} ******".format(args.task))
    logger.info("****** INPUT PATH : {} ******".format(csv_path))
    logger.info("******\n PROMPT : {} ".format(inference_template))

    for index, (shot, column) in zip(indexs, shot2column.items()):
        logger.info("****** INDEX : {}/{} ******".format(indexs, index))
        logger.info("****** SHOT2COLUM : {}/{}/{} ******".format(shot2column, shot, column))

        rows = read_csv(file_path=csv_path, current_len=column, model_name="{} {} shot".format(args.MODEL_NAME, shot))
        num_batches = (len(rows) + args.BATCH_SIZE - 1) // args.BATCH_SIZE

        for batch_idx in range(num_batches):
            batch_data = rows[batch_idx * args.BATCH_SIZE + 1:(batch_idx + 1) * args.BATCH_SIZE + 1]
            if len(batch_data) == 0 or batch_data[-1][column] != "":
                continue

            new_batch_data = []
            for row in batch_data:
                content = row[1].split("\n")
                if index[0] == index[1]:
                    prompt = inference_template.replace("Examples for Guidance:\n{}\n", "").format(content[-2].replace("Review: ", ""))
                else:
                    prompt = inference_template.format("\n".join(content[index[0]:index[1]]), content[-2].replace("Review: ", ""))
                new_batch_data.append(prompt)
            
            infers = in_context_learning_inference(LLM_obeject, sampling_params, new_batch_data)
            for idx in range(len(batch_data)):
                rows[batch_idx * args.BATCH_SIZE + 1 + idx][column] = infers[idx]

            write_csv(csv_path, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM In-Context Learning")

    parser.add_argument("--MODEL_NAME", type=str, required=True)
    parser.add_argument("--BATCH_SIZE", default=8, type=int)
    parser.add_argument("--task", type=str, required=True,)
    parser.add_argument("--data_dir", type=str, required=True,)
    parser.add_argument("--domain", type=str, required=True)

    parser.add_argument("--indexs", type=str, required=True, help="List of index tuples as string")
    parser.add_argument("--shot2column", type=str, required=True, help="Mapping of shots to columns as string")
    args = parser.parse_args()

    indexs = ast.literal_eval(args.indexs) 
    shot2column = {int(key):int(value) for key, value in (item.split(":") for item in args.shot2column.split(","))} 

    if args.task == "triplet":
        inference_template = triplet_inference_template
    elif args.task == "quadruple":
        txt_path = os.path.join(args.data_dir, args.domain, "categories.txt")
        with open(txt_path, "r") as file:
            word_list = [line.strip() for line in file if line.strip()]
        formatted_string = "".join([f"   - {word}\n" for word in word_list])
        inference_template = re.sub(r"\{\}", formatted_string, quadruple_inference_template, count=1)
    
    main(args, inference_template, indexs, shot2column)
