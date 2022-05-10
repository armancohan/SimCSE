import argparse
import io
import logging
import os
import sys

import numpy as np
import torch
import transformers
from prettytable import PrettyTable
from transformers import AutoConfig, AutoModel, AutoTokenizer

from simcse.models import T5ForCL
from train import OurTrainingArguments

# Set up logger
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    parser.add_argument("--model_type", type=str, choices={"bert", "t5"}, default="bert")
    parser.add_argument("--tokenizer_name_or_path", type=str, help="Transformers' tokenizer name or path")
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls",
        help="Which pooler to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest"],
        default="test",
        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        choices=["sts", "transfer", "full", "na"],
        default="sts",
        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "MR",
            "CR",
            "MPQA",
            "SUBJ",
            "SST2",
            "TREC",
            "MRPC",
            "SICKRelatedness",
            "STSBenchmark",
        ],
        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden",
    )

    args = parser.parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    # Load transformers' model checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args.model_type == "bert":
        model = AutoModel.from_pretrained(args.model_name_or_path)
    else:
        # have to load config first, otherwise pad_token_id will be be lost from kwargs
        # this is because the from_pretrained function does not pass arguments that exist in the config
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        # The trainer doesn't save this, so we have to manually set it
        class Dict2Class(object):
            def __init__(self, my_dict):

                for key in my_dict:
                    setattr(self, key, my_dict[key])

        model_args = {
            "temp": 0.05,
            "pooler_type": "cls",
            "do_mlm": False,
            "mlm_weight": 0.1,
            "mlp_only_train": False,
            "hard_negative_weight": 0,
        }
        model = T5ForCL.from_pretrained(
            args.model_name_or_path,
            config=config,
            model_args=Dict2Class(model_args),
            cls_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up the tasks
    if args.task_set == "sts":
        args.tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
    elif args.task_set == "transfer":
        args.tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "full":
        args.tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
        args.tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

    # Set params for SentEval
    if args.mode == "dev" or args.mode == "fasttest":
        # Fast mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}
    elif args.mode == "test":
        # Full mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
        params["classifier"] = {"nhid": 0, "optim": "adam", "batch_size": 64, "tenacity": 5, "epoch_size": 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode("utf-8") for word in s] for s in batch]

        sentences = [" ".join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences, return_tensors="pt", padding=True, max_length=max_length, truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,
            )

        # if isinstance(model, transformers.models.t5.modeling_t5.T5Model):
        #     # prepare input
        #     dtype = batch["input_ids"].dtype
        #     pad_decoder_tokens = torch.full(
        #         size=(batch["input_ids"].shape[0], batch["input_ids"].shape[1]),
        #         fill_value=tokenizer.pad_token_type_id,
        #         device=device,
        #         dtype=dtype,
        #     )
        #     batch["decoder_input_ids"] = pad_decoder_tokens

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            if args.model_type == "t5":
                outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            else:
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
            # if isinstance(model, transformers.models.t5.modeling_t5.T5Model):
            #     assert args.pooler == "cls"
            #     last_hidden = outputs.decoder_hidden_states[-1]
            #     pooler_output = last_hidden[:, 0, :]
            #     hidden_states = None
            # else:
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if args.pooler == "cls":
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == "cls_before_pooler":
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return (
                (last_hidden * batch["attention_mask"].unsqueeze(-1)).sum(1) / batch["attention_mask"].sum(-1).unsqueeze(-1)
            ).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch["attention_mask"].unsqueeze(-1)).sum(1) / batch[
                "attention_mask"
            ].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch["attention_mask"].unsqueeze(-1)).sum(
                1
            ) / batch["attention_mask"].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == "dev":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ["STSBenchmark", "SICKRelatedness"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["devacc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == "test" or args.mode == "fasttest":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]:
            task_names.append(task)
            if task in results:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    scores.append("%.2f" % (results[task]["all"]["spearman"]["all"] * 100))
                else:
                    scores.append("%.2f" % (results[task]["test"]["spearman"].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["acc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
