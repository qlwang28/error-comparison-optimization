# coding=utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../../")
import csv
import re
import ast
import logging
import json
import random
import warnings
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import PeftConfig, get_peft_model, PeftModelForCausalLM, LoraConfig


warnings.simplefilter("ignore", UserWarning)

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

Task for Completion:
Review: "{}"

Output Format: Please provide your analysis in the following JSON format:
{{
    "Output": "[
        ['Aspect1', 'Sentiment1'],
        ...
    ]"
}}
Only output this JSON."""


triplet_inference_template = \
"""\
Task Description:  
Perform Aspect-Based Sentiment Analysis (ABSA) on customer reviews. ABSA involves identifying specific attributes of a product (aspects) discussed in the review, analyzing the sentiment expressed towards these aspects, and extracting any opinion terms used to describe them.

Instructions:  
1. Identify Aspects: Identify all the distinct features or attributes mentioned in the review. Aspects should be extracted as noun phrase spans. If the aspect is implicit (not explicitly mentioned in the review), output `null` for the aspect.
2. Extract Opinions: Identify the explicit opinion terms associated with each aspect. Opinions should be extracted as noun phrase spans. If the opinion term is implicit (not explicitly mentioned in the review), output `null` for the opinion term.
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

Task for Completion:
Review: "{}"

Output Format: Please provide your analysis in the following JSON format:
{{
    "Output": "[
        ['Aspect1', 'Opinion1', 'Sentiment1'],
        ...
    ]"
}}
Only output this JSON."""


quadruple_inference_template = \
"""\
Task Description:
Perform Aspect-Based Sentiment Analysis (ABSA) on customer reviews. ABSA aims to predict all quads (aspect term, opinion term, aspect category, sentiment polarity) for a given review.

Instructions:
1. Identify Aspects: Identify all the distinct features or attributes mentioned in the review. Aspects should be extracted as noun phrase spans. If the aspect is implicit (not explicitly mentioned in the review), output `null` for the aspect.
2. Extract Opinion Terms: Identify the specific words or phrases that convey the sentiment towards each aspect. Opinions should be extracted as noun phrase spans. If the opinion term is implicit (not explicitly mentioned in the review), output `null` for the opinion term.
3. Determine Aspect Categories: Map each identified aspect to a predefined category. It consists of an entity label and attribute label, with possible values like: 
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

Task for Completion:
Review: "{}"

Output Format: Please provide your analysis in the following JSON format:
{{
    "Output": "[
        ['Aspect1', 'Opinion1', 'Aspect category1', 'Sentiment1'],
        ...
    ]"
}}
Only output this JSON."""



class DataCollatorForPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        keys = ['input_ids', 'attention_mask', 'labels', 'good_input_ids', 'good_attention_mask', 'good_labels', 'bad_input_ids', 'bad_attention_mask', 'bad_labels', 'score']
        batch_data = {key: [] for key in keys}
        max_lengths = {
            key: max([len(example[key]) for example in examples]) for key in keys
        }

        for example in examples:
            for key in keys:
                padding_length = max_lengths[key] - len(example[key])
                if 'input_ids' in key:
                    padded = padding_length * [self.tokenizer.pad_token_id] + example[key]
                elif 'attention_mask' in key:
                    padded = padding_length * [0] + example[key]
                elif 'labels' in key:
                    padded = padding_length * [-100] + example[key]
                elif 'score' == key:
                    padded = example[key]
                else:
                    raise ValueError
                batch_data[key].append(padded)

        for key in keys:
            if key != 'score':
                batch_data[key] = torch.tensor(batch_data[key], dtype=torch.long)
            else:
                batch_data[key] = torch.tensor(batch_data[key], dtype=torch.float)
        return batch_data


class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, src_texts, tgt_texts, good_tgts, bad_tgts, scores):
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.good_tgts = good_tgts
        self.bad_tgts = bad_tgts
        self.scores = scores

    def __len__(self):
        return len(self.src_texts)

    def Tokenizer(self, tgt_text, text_encodings):
        label_encodings = self.tokenizer.encode(tgt_text + self.tokenizer.eos_token, add_special_tokens=False)
        input_ids = text_encodings + label_encodings
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(text_encodings) + label_encodings
        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        good_tgt = self.good_tgts[idx]
        bad_tgt = self.bad_tgts[idx]
        score = self.scores[idx]

        text_encodings = self.tokenizer.encode(self.tokenizer.bos_token + src_text, add_special_tokens=False)
        input_ids, attention_mask, labels = self.Tokenizer(tgt_text=tgt_text, text_encodings=text_encodings)
        good_input_ids, good_attention_mask, good_labels = self.Tokenizer(tgt_text=good_tgt, text_encodings=text_encodings)
        bad_input_ids, bad_attention_mask, bad_labels = self.Tokenizer(tgt_text=bad_tgt, text_encodings=text_encodings)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'good_input_ids': good_input_ids,
            'good_attention_mask': good_attention_mask,
            'good_labels': good_labels,
            'bad_input_ids': bad_input_ids,
            'bad_attention_mask': bad_attention_mask,
            'bad_labels': bad_labels,
            'score':score,           
        }
        return item


class LLM_Instance(object):
    def __init__(self, model_path, generated_length):
        super(LLM_Instance, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer.pad_token = "<|reserved_special_token_1|>"
        self.generated_length = generated_length

        self.lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,           # rank of the low-rank adaptation matrices
            bias="none",
            lora_alpha=32,      # scaling factor
            lora_dropout=0.1,   # dropout rate
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )

    def read_csv(self, file_path, column, column_name):
        lines = []
        with open(file_path, "r", encoding='utf-8') as file_obj:
            reader_obj = csv.reader(file_obj)
            for i, row in enumerate(reader_obj):
                if row[0] == "":
                    continue
                if len(row) == column:
                    row.insert(column, column_name if i == 0 else "")
                lines.append(row)
        return lines

    def update_csv(self, file_path, rows):
        with open(file_path, "w", encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(rows)

    def decode_generate(self, texts, decoded_times):
        encoded_inputs = self.tokenizer(texts, padding='longest', return_tensors='pt')
        input_ids = encoded_inputs['input_ids']
        leng = input_ids.shape[1]
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        with torch.no_grad():
            greedy_outputs = self.llm_obeject.generate(input_ids=input_ids.cuda(),
                attention_mask=encoded_inputs['attention_mask'].cuda(),
                max_new_tokens=self.generated_length,
                do_sample=False,
                pad_token_id=pad_id,
            )
            decoded_outputs = [self.tokenizer.decode(output[leng:], skip_special_tokens=True) for output in greedy_outputs]
            all_outputs = [decoded_outputs]

            if decoded_times > 1:
                num_return_sequences = decoded_times - 1
                beam_outputs = self.llm_obeject.generate(input_ids=input_ids.cuda(),
                    attention_mask=encoded_inputs['attention_mask'].cuda(),
                    max_new_tokens=self.generated_length,
                    do_sample=False,
                    num_beam_groups=6,
                    num_beams=6,
                    diversity_penalty=3.0,
                    early_stopping=True,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=pad_id,
                )
                outputs = [self.tokenizer.decode(output[leng:], skip_special_tokens=True) for output in beam_outputs]
                all_outputs.extend([outputs[i::num_return_sequences] for i in range(num_return_sequences)])

        return all_outputs

    def in_context_learning(self, prompts, decoded_times=1):
        response_list = self.decode_generate(prompts, decoded_times)
        results = []
        for i, _ in enumerate(prompts):
            output = []
            for j in range(decoded_times):
                resp = re.sub(r'\s+', ' ', response_list[j][i].lower().strip())
                resp = re.sub(r'\[\s+\[', '[[', resp)
                resp = re.sub(r'\]\s+\]', ']]', resp)
                resp = resp.replace("[{", "[").replace("}]", "]")
                resp = resp.replace("[(", "[[").replace(")]", "]]").replace('["["', '["')
                
                try:
                    json_obj = json.loads(re.findall(r'\{[^{}]*\}', resp)[0])["output"]
                except Exception:
                    json_obj = (
                        re.search(r'"output":\s*(\[\[.*?\]\])', resp) or 
                        re.search(r'(\[\[.*?\]\])', resp) or 
                        re.search(r'"output":\s*(\[[^\]]*\])', resp) or 
                        re.search(r'(\[[^\]]*\])', resp)
                    )

                    if json_obj:
                        json_obj = json_obj.group(1)
                        if not json_obj.startswith("[["):
                            json_obj = "[[" + json_obj[1:-1] + "]]"
                    else:
                        json_obj = re.sub(r'\n+', '\n', resp.strip())

                output.append(str(json_obj))

            results.append('\n'.join(output))
        return results
    
    def inference(self, file_path, column, column_name, batch_size, decoded_times=1):
        rows = self.read_csv(file_path=file_path, column=column, column_name=column_name)
        num_batches = (len(rows) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc='In context learning:'):
            batch_data = rows[batch_idx * batch_size + 1 : (batch_idx + 1) * batch_size + 1]
            if len(batch_data) == 0 or batch_data[-1][column] != "":
                continue
            
            prompts = []
            for row in batch_data:
                content = row[1].split("\n")
                if decoded_times == 1:
                    prompt = self.args.inference_template.format("\n".join(content[self.args.start_index:self.args.end_index]), content[-2].replace("Review: ", ""))
                else:
                    prompt = self.args.inference_template.format("\n".join(content[self.args.start_index + 4:self.args.end_index]), content[-2].replace("Review: ", ""))
                prompts.append(prompt)

            infers = self.in_context_learning(prompts, decoded_times)
            for idx in range(len(batch_data)):
                rows[batch_idx * batch_size + 1 + idx][column] = infers[idx]

            self.update_csv(file_path, rows)
            torch.cuda.empty_cache()
        logger.info("Done in context learning in %s file ......" % file_path)

    def identifying_predictions(self, label, predictions):
        def calculate_boundary_penalty(pred_txt, lab_txt, penalty1, penalty2):
            penalty = 0
            if pred_txt.startswith(lab_txt) or pred_txt.endswith(lab_txt):
                penalty -= penalty1
            elif lab_txt.endswith(pred_txt) or (pred_txt in lab_txt and not lab_txt.startswith(pred_txt) and not lab_txt.endswith(pred_txt)): 
                penalty -= penalty1
            elif lab_txt.startswith(pred_txt): 
                penalty -= penalty2
            return penalty

        label = ast.literal_eval(label.lower())
        tmps = []
        for pred in predictions:
            try:
                tmp = ast.literal_eval(pred.lower())  
                if tmp == [[]]: 
                    tmp = []
            except (ValueError, SyntaxError) as e: 
                if self.args.task == "triplet":
                    tmp = [['none', 'none', 'none']] 
                elif self.args.task == "quadruple":
                    tmp = [['none', 'none', 'none', 'none']]
                else:
                    tmp = [['none', 'none']]
            tmps.append(tmp)

        scores = []
        label_set = set(tuple(item) for item in label)
        for prediction in tmps:
            prediction_set = set(tuple(item) for item in prediction)
            match_score = 2 * len(label_set & prediction_set) 
            length_penalty = abs(len(prediction) - len(label))  
            adjusted_score = match_score - length_penalty

            for pred in prediction:
                if self.args.task == "triplet" and (not isinstance(pred, list) or len(pred) != 3): 
                    pred = ['none', 'none', 'none']
                if self.args.task == "quadruple" and (not isinstance(pred, list) or len(pred) != 4):
                    pred = ['none', 'none', 'none', 'none']

                for lbl in label:
                    pred_apt, pred_sent, lab_apt, lab_sent = pred[0], pred[-1], lbl[0], lbl[-1]
                    pred_opin, lab_opin, pred_cate, lab_cate = None, None, None, None
                    if self.args.task == "triplet":
                        pred_opin, lab_opin = pred[1], lbl[1]
                    if self.args.task == "quadruple":
                        pred_cate, lab_cate = pred[2], lbl[2]
                    
                    if pred_apt != lab_apt and pred_sent == lab_sent:
                        adjusted_score += calculate_boundary_penalty(pred_txt=pred_apt, lab_txt=lab_apt, penalty1=0.25, penalty2=0.5)
                    if pred_opin and lab_opin and pred_opin != lab_opin:
                        adjusted_score += calculate_boundary_penalty(pred_txt=pred_opin, lab_txt=lab_opin, penalty1=0.25, penalty2=0.5)
                    if pred_apt != lab_apt and pred_sent != lab_sent:
                        adjusted_score += calculate_boundary_penalty(pred_txt=pred_apt, lab_txt=lab_apt, penalty1=0.75, penalty2=0.75)
                    if pred_cate and lab_cate and pred_apt == lab_apt and pred_cate != lab_cate:
                        adjusted_score -= 0.25
                    if pred_apt == lab_apt and pred_sent != lab_sent:
                        if ('pos' in lab_sent and 'neg' in pred_sent) or ('neg' in lab_sent and 'pos' in pred_sent):
                            adjusted_score -= 1
                        else:
                            adjusted_score -= 0.5
            scores.append(adjusted_score)

        best_id = scores.index(max(scores))
        worst_id = scores.index(min(scores))
        return predictions[best_id], predictions[worst_id], scores

    def load_ABSAData(self, file_path):
        src_texts, tgt_texts, good_tgts, bad_tgts, scores = [], [], [], [], []
        label_template = """{{\n    "Output": "{}" \n}}"""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row[1].split("\n")) == 0:
                    continue
                text = row[1].split("\n")[-2].replace("Review: ", "")
                if len(text.split()) > self.args.max_seq_length:
                    continue
                label = row[2].strip()

                src = self.args.inference_template.replace("\nExamples for Guidance:\n{}\n", "").format(text)
                tgt = label_template.format(label)
                src_texts.append(src)
                tgt_texts.append(tgt)

                if self.args.preference_type != "SFT": 
                    prediction_list = row[-1].strip().split("\n")
                    good_tgt, bad_tgt, score = self.identifying_predictions(label=label, predictions=prediction_list)
                    # good_tgt, bad_tgt = random.sample(prediction_list, 2)
                else:
                    good_tgt, bad_tgt, score = "love", "hate", [1.0, 1.2]

                good_tgts.append(label_template.format(good_tgt))
                bad_tgts.append(label_template.format(bad_tgt))
                scores.append(score)
        return src_texts, tgt_texts, good_tgts, bad_tgts, scores

    def masked_log_probs(self, log_probs, labels):
        mask = (labels != -100)
        clone_labels = labels.clone() 
        clone_labels[~mask] = 0

        log_probs = log_probs.gather(2, clone_labels.unsqueeze(-1)).squeeze(-1)

        valid_log_probs = log_probs * mask.float()
        sum_log_probs = valid_log_probs.sum(dim=1)
        count_valid = mask.sum(dim=1).float() 

        average_log_probs = sum_log_probs / count_valid.clamp(min=1)
        return average_log_probs

    def lora_training(self, peft_model, file_path, 
        num_epochs=3,
        batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        ):
        src_texts, tgt_texts, good_tgts, bad_tgts, scores = self.load_ABSAData(file_path)
        dataset = ABSADataset(self.tokenizer, src_texts, tgt_texts, good_tgts, bad_tgts, scores)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size, collate_fn=DataCollatorForPadding(tokenizer=self.tokenizer))
        num_optimization_steps = num_epochs * len(dataloader)
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(dataset))
        logger.info("Total optimization steps = %d", num_optimization_steps)
        logger.info("***** Example *****")
        logger.info("Source text: \n%s ", src_texts[1])
        logger.info("Target text: \n%s ", tgt_texts[1])
        logger.info("Good Target text: \n%s ", good_tgts[1])
        logger.info("Bad Target text: \n%s ", bad_tgts[1])
        logger.info("Sorce : \n{} ".format(scores[1]))

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_optimization_steps)
        peft_model.zero_grad()
        peft_model.train()
        global_step = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                peft_model.zero_grad()
                batch = tuple(t.cuda() for t in batch.values())

                if self.args.preference_type == "SFT":
                    input_ids, attention_mask, labels, _, _, _, _, _, _, _ = batch
                    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                if self.args.preference_type == "DPO":
                    loss = self.dpo_algorithm(peft_model, self.ref_obeject, batch)
                if self.args.preference_type == "OUR":
                    loss = self.calibration_algorithm(peft_model, batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 10)
                optimizer.step()
                scheduler.step()

                if global_step % 50 == 0:
                    logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}".format(epoch, global_step, num_optimization_steps, loss.item()))
                global_step += 1

    def dpo_algorithm(self, peft_model, ref_obeject, batch, beta = 1.0):
        input_ids, attention_mask, labels, _, _, _, bad_input_ids, bad_attention_mask, bad_labels, _ = batch

        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        bads = peft_model(input_ids=bad_input_ids, attention_mask=bad_attention_mask, labels=bad_labels)
        ref_outputs = ref_obeject(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ref_bads = ref_obeject(input_ids=bad_input_ids, attention_mask=bad_attention_mask, labels=bad_labels)

        pi_yw_logprobs = F.log_softmax(outputs.logits, dim=-1)
        pi_yl_logprobs = F.log_softmax(bads.logits, dim=-1)
        ref_yw_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
        ref_yl_logprobs = F.log_softmax(ref_bads.logits, dim=-1)

        pi_logratios = self.masked_log_probs(log_probs=pi_yw_logprobs, labels=labels) - self.masked_log_probs(log_probs=pi_yl_logprobs, labels=bad_labels)
        ref_logratios = self.masked_log_probs(log_probs=ref_yw_logprobs, labels=labels) - self.masked_log_probs(log_probs=ref_yl_logprobs, labels=bad_labels)
        
        dpo_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
        loss = outputs.loss + 0.25 * dpo_loss.mean() 
        return loss

    def calibration_algorithm(self, peft_model, batch, beta = 0.1):
        input_ids, attention_mask, labels, good_input_ids, good_attention_mask, good_labels, bad_input_ids, bad_attention_mask, bad_labels, _ = batch

        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        good_opt = peft_model(input_ids=good_input_ids, attention_mask=good_attention_mask, labels=good_labels)
        good_loss = good_opt.loss
        bad_opt = peft_model(input_ids=bad_input_ids, attention_mask=bad_attention_mask, labels=bad_labels)
        bad_loss = bad_opt.loss

        ranking_loss = torch.clamp(beta - bad_loss + good_loss, min=0) + torch.clamp(beta - good_loss + outputs.loss, min=0)
        loss = outputs.loss + 0.25 * ranking_loss
        # ranking_loss = torch.clamp(beta - bad_loss + outputs.loss, min=0)
        return loss

    def optimize_weights(self, model_path, file_path, output_dir, epoch, batch_size, learning_rate):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        logger.info("****** Print its Architecture to see all module names ****** ")
        print(model)
        peft_model = get_peft_model(model, self.lora_config)
        peft_model.print_trainable_parameters()

        self.lora_training(peft_model=peft_model,
            file_path=file_path,
            num_epochs=epoch, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            )
        peft_model.save_pretrained(output_dir, save_lora_only=True)

        
if __name__ == '__main__':
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Run LLM SFT")
    parser.add_argument("--MODEL_NAME", type=str, required=True)
    parser.add_argument("--TRAIN_BATCH_SIZE", default=1, type=int)
    parser.add_argument("--EPOCH", default=2, type=int)
    parser.add_argument("--ITER", default=1, type=int)
    parser.add_argument("--TEST_BATCH_SIZE", default=20, type=int)
    parser.add_argument("--NUM_OUTPUTS", default=5, type=int)

    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--preference_type", default="SFT", type=str)
    parser.add_argument("--task", type=str, required=True,)
    parser.add_argument("--data_dir", type=str, required=True,)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--max_seq_length", default=60, type=int, help="The maximum total input sequence length")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.", default=False)
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.", default=False)
    parser.add_argument("--train_column", default=-1, type=int)
    parser.add_argument("--test_column", default=-1, type=int)
    args = parser.parse_args()

    if args.task == "triplet":
        args.inference_template = triplet_inference_template
        args.start_index = 5
        args.end_index = 13
    if args.task == "quadruple":
        txt_path = os.path.join(args.data_dir, args.domain, "categories.txt")
        with open(txt_path, "r") as file:
            word_list = [line.strip() for line in file if line.strip()]
        formatted_string = "".join([f"   - {word}\n" for word in word_list])
        args.inference_template = re.sub(r"\{\}", formatted_string, quadruple_inference_template, count=1)
        args.start_index = 6
        args.end_index = 14

    model_path = 'models/pre-trained_model/{}'.format(args.MODEL_NAME)
    llm_instance = LLM_Instance(model_path, generated_length=512)
    llm_instance.args = args
    train_path = os.path.join(args.data_dir, args.domain, "reason_train_bm25_16_shot.csv")
    test_path = os.path.join(args.data_dir, args.domain, "reason_test_bm25_16_shot.csv")
    output_dir = os.path.join(args.data_dir, args.domain, "{}_adapter_preference_{}".format(args.MODEL_NAME, args.preference_type))
    logger.info("****** TASK : {} ******".format(args.task))
    logger.info("****** Preference Optimization : {} ******".format(args.preference_type))
    logger.info("****** Train INPUT PATH : {} ******".format(train_path))
    logger.info("****** Test INPUT PATH : {} ******".format(test_path))
    logger.info("****** OUTPUT PATH : {} ******".format(output_dir))
    logger.info("******\n PROMPT : {} ".format(args.inference_template))
    
    if args.do_train:
        logger.info("******************** Start {} Training Process ********************".format(args.preference_type))
        if args.preference_type in ["PPO"]:
            ref_dir = os.path.join(args.data_dir, args.domain, "{}_adapter_preference_{}".format(args.MODEL_NAME, "SFT"))
            logger.info("******  Reference Dir: {} ******".format(ref_dir))
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
            llm_instance.ref_obeject = PeftModelForCausalLM.from_pretrained(model, ref_dir)
            llm_instance.ref_obeject.eval()
            for param in llm_instance.ref_obeject.parameters():
                param.requires_grad = False

        if not os.path.isdir(output_dir):
            llm_instance.optimize_weights(model_path=model_path,
                file_path=train_path,
                output_dir=output_dir,
                epoch=args.EPOCH, 
                batch_size=args.TRAIN_BATCH_SIZE, 
                learning_rate=args.learning_rate,
                )
        logger.info("******************** End Optimize Weights process ********************")

    if args.do_eval:
        logger.info("******************** Start {} Testing Process ********************".format(args.preference_type))
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        llm_instance.llm_obeject = PeftModelForCausalLM.from_pretrained(model, output_dir)
        llm_instance.llm_obeject.eval()

        if args.preference_type == "SFT":
            llm_instance.inference(file_path=train_path, 
                column=args.train_column, 
                column_name="{}".format(args.MODEL_NAME), 
                batch_size=args.TEST_BATCH_SIZE,
                decoded_times=args.NUM_OUTPUTS,
                )

        llm_instance.inference(file_path=test_path, 
            column=args.test_column, 
            column_name="{} {}".format(args.MODEL_NAME, args.preference_type), 
            batch_size=args.TEST_BATCH_SIZE,
            )
        
