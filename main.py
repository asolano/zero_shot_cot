import argparse
import logging
from re import M
import torch
import random
import time
import os
from utils import *
import re


def prepare_question_template(data, demo, args):
    x, y = data

    x = [f"Q: {q}\nA:" for q in x]
    y = [a.strip() for a in y]

    if args.method == "zero_shot":
        x = [f"{a} {args.direct_answer_trigger_for_zeroshot}" for a in x]
    elif args.method == "zero_shot_cot":
        x = [f"{a} {args.cot_trigger}" for a in x]
    elif args.method == "few_shot":
        x = [f"{demo}{a}" for a in x]
    elif args.method == "few_shot_cot":
        x = [f"{demo}{a}" for a in x]
    else:
        raise ValueError("method is not properly defined ...")

    return x, y


def remove_redundant_pattern(s, pattern):
    found = list(re.finditer(pattern, s))
    if len(found) > 0:
        index = found[0].span()[0]
        s = s[:index]
    return s


def batch_work(i, data, demo, decoder, args):
    print("*" * 20)
    print(f"batch {i+1}")

    prompts, true_answers = prepare_question_template(data, demo, args)

    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
    
    predictions = decoder.decode(args, prompts, max_length, i, 1)

    # After here z should not contain any Q: or A: elements
    predictions = [ remove_redundant_pattern(p, 'Q:') for p in predictions ] 
    predictions = [ remove_redundant_pattern(p, 'A:') for p in predictions ] 

    # Answer extraction for zero-shot-cot ...
    if args.method == "zero_shot_cot":
        # Do a second prediction with the answer trigger
        max_length = args.max_length_direct

        prompts_cot = [f"{p1}{p2} {args.direct_answer_trigger_for_zeroshot_cot}" for (p1, p2) in zip(prompts, predictions)]
        predictions_cot = decoder.decode(args, prompts_cot, max_length, i, 2)

        print(f'>>>\nDEBUG predictions_cot[0]: {predictions_cot[0]}')

        # Remove the Q: A: here too
        predictions_cot = [ remove_redundant_pattern(s, 'Q:') for s in predictions_cot ] 
        predictions_cot = [ remove_redundant_pattern(s, 'A:') for s in predictions_cot ]

        print(f'DEBUG predictions_cot[0] (after filter): {predictions_cot[0]}\n<<<\n')

        prompts = prompts_cot
        predictions = predictions_cot

    # Log predictions
    logs = []
    for i, p in enumerate(predictions):
        logs.append(f'{prompts[i]}{p}')

    # Extract answer
    answers = [answer_cleansing(args, p) for p in predictions]

    # Choose the most frequent answer from the list ...
    new_correct = []
    for i, a in enumerate(answers):
        logs[i] += "\n" + "*" * 20 + "\n"
        logs[i] += f"prediction={a}\n"
        logs[i] += f"truth={true_answers[i]}\n"
        logs[i] += "*" * 20 + "\n"

        # Checking answer ...
        # correct = (np.array([pred]) == np.array([y])).sum().item()
        correct = (np.array([a]) == np.array([true_answers[i]])).sum().item()
        # correct_list.append(correct)
        # total += 1  # np.array([y]).size(0)
        new_correct.append(correct)

    for entry in logs:
        print(entry, flush=True)

    return new_correct


def main():
    args = parse_arguments()
    print("*****************************")
    print(args)
    print("*****************************")

    if "gpt" in args.model or "api" in args.model:
        # FIXME change minibatch_size to 1 automaticaly?
        assert args.minibatch_size == 1, "APIs do not currently accept batched inputs"

    fix_seed(args.random_seed)

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    print("setup data loader ...")

    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        demo = None

    total = 0
    correct_list = []
    for i, data in enumerate(dataloader):
        new_correct = batch_work(i, data, demo, decoder, args)
        correct_list.extend(new_correct)
        total += len(new_correct)
        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name",
        type=str,
        default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]",
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset",
        type=str,
        default="aqua",
        choices=[
            "aqua",
            "gsm8k",
            "commonsensqa",
            "addsub",
            "multiarith",
            "strategyqa",
            "svamp",
            "singleeq",
            "bigbench_date",
            "object_tracking",
            "coin_flip",
            "last_letters",
        ],
        help="dataset used for experiment",
    )

    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request",
    )

    parser.add_argument(
        "--max_num_worker",
        type=int,
        default=3,
        help="maximum number of workers for dataloader",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt3",
        choices=[
            "gpt3",
            "gpt3-medium",
            "gpt3-large",
            "gpt3-xl",
            "bloom-1b7",
            "bloom-3b",
            "bloom-7b1",
            "bloom",
            "bloom-api",
            "opt",
            "flan-t5-xxl"
        ],
        help="model used for decoding. Note that 'gpt3' are the smallest models.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="zero_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"],
        help="method",
    )
    parser.add_argument(
        "--cot_trigger_no",
        type=int,
        default=1,
        help="A trigger sentence that elicits a model to execute chain of thought",
    )
    parser.add_argument(
        "--max_length_cot",
        type=int,
        default=128,
        help="maximum length of output tokens by model for reasoning extraction",
    )
    parser.add_argument(
        "--max_length_direct",
        type=int,
        default=32,
        help="maximum length of output tokens by model for answer extraction",
    )
    parser.add_argument(
        "--limit_dataset_size",
        type=int,
        default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.",
    )
    parser.add_argument("--api_time_interval", type=float, default=1.0, help="")
    parser.add_argument("--log_dir", type=str, default="./log/", help="log directory")
    parser.add_argument(
        "--int8", action="store_true", help="Use bitsandbytes int8 optimizers (Bloom)"
    )

    args = parser.parse_args()

    datasets = {
        "aqua": dict(
            path="./dataset/AQuA/test.json",
            trigger="\nTherefore, among A through E, the answer is",
            plausible_trigger=None,
        ),
        "gsm8k": dict(
            path="./dataset/grade-school-math/test.jsonl",
            trigger="\nTherefore, among A through E, the answer is",
            plausible_trigger=None,
        ),
        "commonsensqa": dict(
            path="./dataset/CommonsenseQA/dev_rand_split.jsonl",
            trigger="\nTherefore, among A through E, the answer is",
            plausible_trigger="Choose the most plausible answer from among choices A through E.",
        ),
        "addsub": dict(
            path="./dataset/AddSub/AddSub.json",
            trigger="\nTherefore, the answer (arabic numerals) is",
            plausible_trigger=None,
        ),
        "multiarith": dict(
            path="./dataset/MultiArith/MultiArith.json",
            trigger="\nTherefore, the answer (arabic numerals) is",
            plausible_trigger=None,
        ),
        "strategyqa": dict(
            path="./dataset/StrategyQA/task.json",
            trigger="\nTherefore, the answer (Yes or No) is",
            plausible_trigger=None,
        ),
        "svamp": dict(
            path="./dataset/SVAMP/SVAMP.json",
            trigger="\nTherefore, the answer (arabic numerals) is",
            plausible_trigger=None,
        ),
        "singleeq": dict(
            path="./dataset/SingleEq/questions.json",
            trigger="\nTherefore, the answer (arabic numerals) is",
            plausible_trigger=None,
        ),
        "bigbench_date": dict(
            path="./dataset/Bigbench_Date/task.json",
            trigger="\nTherefore, among A through F, the answer is",
            plausible_trigger=None,
        ),
        "object_tracking": dict(
            path="./dataset/Bigbench_object_tracking/task.json",
            trigger="\nTherefore, among A through C, the answer is",
            plausible_trigger=None,
        ),
        "coin_flip": dict(
            path="./dataset/coin_flip/coin_flip.json",
            trigger="\nTherefore, the answer (Yes or No) is",
            plausible_trigger=None,
        ),
        "last_letters": dict(
            path="./dataset/last_letters/last_letters.json",
            trigger="\nTherefore, the answer is",
            plausible_trigger=None,
        ),
    }
    try:
        data = datasets[args.dataset]

        abs_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(abs_path, data["path"])
        args.dataset_path = path

        args.direct_answer_trigger = data["trigger"]
        args.plausible_answer_trigger = data["plausible_trigger"]
    except KeyError:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    cot_triggers = {
        1: "Let's think step by step.",
        #2: "We should think about this step by step.",
        3: "First,",
        4: "Before we dive into the answer,",
        #5: "Proof followed by the answer.",
        #6: "Let's think step by step in a realistic way.",
        #7: "Let's think step by step using common sense and knowledge.",
        8: "Let's think like a detective step by step.",
        9: "Let's think about this logically.",
        #10: "Let's think step by step. First,",
        11: "Let's think",
        12: "Let's solve this problem by splitting it into steps.",
        13: "The answer is after the proof.",
        14: "Let's be realistic and think step by step.",
        # From meta reviewer
        15: "Let's solve the problem bit by bit.",
        16: "If we solve the problem stepwise,",
        17: "If we solve the problem gradually,",
        18: "If we solve the problem piecemeal,",
        19: "Let's be smart about this.",
        20: "Let's be careful we get the right answer.",
        21: "Let's make sure we get the right answer.",
        # From GPT-3 
        22: "Let's try to understand.",
        23: "Think about it a little more.",
        24: "Think about it in more detail.",
        25: "Let's do it step by step.",
        26: "Let's do it a little at a time.",
        27: "Let's go through it step by step.",
        28: "Let's divide it into parts.",
        29: "Let's take it slow.",
    }
    try:
        args.cot_trigger = cot_triggers[args.cot_trigger_no]
    except KeyError:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
