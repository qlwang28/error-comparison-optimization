# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import codecs
import os
import random
import sys
sys.path.append('../')
sys.path.append("../../")

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from optimization import AdamW, Warmup
from utils import (convert_examples_to_features, EntityProcessor)


logger = logging.getLogger(__name__)
torch.set_num_threads(12)


class BertForSequenceLabeling(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceLabeling, self).__init__(config)
        config.clue_num = 0
        config.num_labels = num_labels
        self.config = config
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.cel = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss = self.cel(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            predicts = torch.argmax(logits, dim=2)
            return  predicts


def train(args, model, tokenizer, train_dataset, eval_dataset):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    num_optimization_steps = len(train_dataloader) * args.num_train_epochs

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Total optimization steps = %d", num_optimization_steps)

    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, _, labels, lens = batch
            loss = model(inputs, attention_mask=masks, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}".format(epoch, global_step, num_optimization_steps, loss.item()))
            
            global_step += 1
            if args.evaluate_during_training and global_step % args.eval_logging_steps == 0: 
                model.eval()
                evaluate(args, model, tokenizer, eval_dataset)
                model.train()
        torch.cuda.empty_cache()


def evaluate(args, model, tokenizer, eval_dataset):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)
    model.eval()
    out_preds, out_label_ids = [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, _, label_ids, lens = batch
            predicts = model(input_ids, attention_mask=input_mask)
            out_preds.append(predicts.detach().cpu().numpy())
            out_label_ids.append(label_ids.detach().cpu().numpy())

    test_result = []
    for numpy_result in out_preds:
        test_result.extend(numpy_result.tolist())
    examples = [exam for k, exam in enumerate(args.test_examples)]
    eval_conlleval(args, examples, tokenizer, test_result, os.path.join(args.data_dir, 'eval_file.txt'))


def load_and_cache_examples(args, tokenizer, dataname="train"):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = args.processor.get_examples(args.data_dir, "train")
    elif dataname == "test":
        examples =args.processor.get_examples(args.data_dir, "test")
        args.test_examples = examples
    else:
        raise ValueError("(evaluate and dataname) parameters error !")

    features = convert_examples_to_features(examples, args.label2id, args.max_seq_length, tokenizer, 
        cls_token = tokenizer.cls_token, 
        sep_token = tokenizer.sep_token,
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        sequence_a_segment_id=0, 
        sequence_b_segment_id=1,
        pad_token_label_id = args.pad_token_label_id,
        )

    all_inputs = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.tokens_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_inputs, all_masks, all_segments, all_labels, all_lens)
    return dataset


def eval_conlleval(args, examples, tokenizer, result, convall_file):
    import traceback
    id2label = {index:label for label, index in args.label2id.items()}
    def test_result_to_pair(writer):
        for example, prediction in zip(examples, result):
            line = ''
            line_token = example.text_a
            line_label = example.label_a
            len_seq = len(line_label)
            if len(line_token) != len(line_label):
                logger.info(example.text_a)
                logger.info(example.label_a)
                break

            step = 0 if bool('xlnet' in args.pretrained_model_name) else 1
            for index in range(len_seq):
                if index >= args.max_seq_length - 2:
                    break
                cur_token = line_token[index]
                cur_label = line_label[index]
                sub_token = tokenizer.tokenize(cur_token)
                try:
                    if len(sub_token) == 0:
                        raise ValueError
                    elif len(sub_token) == 1:                            
                        line += cur_token + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                        step += 1
                    elif len(sub_token) > 1:
                        if cur_label.startswith("B-"):
                            line += sub_token[0] + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1
                            cur_label = "I-" + cur_label[2:]
                            sub_token = sub_token[1:]
                        for t in sub_token:
                            line += t + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1

                except Exception as e:
                    logger.warning(e)
                    logger.warning(example.text_a)
                    logger.warning(example.label_a)
                    line = ''
                    break
            writer.write(line + '\n')

    with codecs.open(convall_file, 'w', encoding='utf-8') as writer:
        test_result_to_pair(writer)
    from conlleval import return_report
    eval_result = return_report(convall_file)
    logger.info(''.join(eval_result))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True)

    parser.add_argument("--eval_results_dir", default="eval_results.txt", type=str, help="Where do you want to store the eval results")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", default=48, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
                        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `'none'`, `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument('--pretrained_vocab', type=str, default='', help="to load pretrain vocab (txt file)")
    parser.add_argument('--pretrained_params', type=str, default='', help='to load pretraining model params')

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--eval_logging_steps', type=int, default=300, help="Log every X evalution steps.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.processor = EntityProcessor()
    args.label2id = args.processor.get_labels()
    logger.info("LABEL : {}".format(args.label2id))
    args.num_labels = len(args.label2id)
    tokenizer = BertTokenizer(args.pretrained_vocab, do_lower_case=args.do_lower_case)

    train_datasets = load_and_cache_examples(args, tokenizer, dataname="train")
    eval_datasets = load_and_cache_examples(args, tokenizer, dataname="test")
    model = BertForSequenceLabeling.from_pretrained(args.pretrained_params, num_labels=args.num_labels)
    model.to(args.device)
    if args.do_train:
        train(args, model, tokenizer, train_datasets, eval_datasets)
    if args.do_eval:
        evaluate(args, model, tokenizer, eval_datasets)


if __name__ == "__main__":
    main()
