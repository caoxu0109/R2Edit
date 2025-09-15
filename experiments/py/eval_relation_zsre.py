"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_forget_quality_zsre(
    #original_model: AutoModelForCausalLM,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )

    # f_subject, f_target_new = (
    #     record["forget_prompts"][x] for x in ["subject", "target_new"]
    # )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    # forget_prompts = record["forget_prompts"]
    # f_p_p = record["forget_paraphrase_prompts"]
    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # forget_probs_prompts = [
    #     forget_prompts,
    #     #f_p_p,
    # ]
    #forget_tok = tok(" " + f_target_new["str"])["input_ids"]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    if 'llama' in model.config._name_or_path.lower():
        target_tok = target_tok[1:]
        #forget_tok = forget_tok[1:]
    #forget_length = len(target_tok)
    inp_prompts_og = list(chain(*prob_prompts))
    #forget_prompts_og = list(chain(*forget_probs_prompts))
    
    inp_prompts = [
        el + tok.decode(target_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    # fog_prompts = [
    #     el + tok.decode(forget_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(forget_tok[:i])
    #     for el in forget_prompts_og
    #     for i in range(len(forget_tok))
    # ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]
    # fog_targets = [
    #     tok.decode(forget_tok[i])
    #     for _ in range(len(forget_prompts_og))
    #     for i in range(len(forget_tok))
    # ]
    #forget_probs = get_orignal_response(original_model,model, tok, fog_prompts)
    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)
    #forget_probs = test_batch_prediction_acc(model, tok, fog_prompts, fog_targets)
    # Predict for neighborhood prompts (dictionary format).
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    # forget_correct = test_batch_prediction_acc(
    # model,
    # tok,
    # [
    #     el["prompt"].format(record["requested_rewrite"])
    #     for el in forget_prompts
    # ],
    # [el["target"] for el in forget_prompts],
    # )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()

    # fog_cutoffs = [0] + np.cumsum(
    #     [l * len(forget_tok) for l in map(len, forget_probs_prompts)]
    # ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    #forget_probs = [forget_probs[fog_cutoffs[i - 1] : fog_cutoffs[i]] for i in range(1, len(fog_cutoffs))]
    all = ret_probs #+ forget_probs
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": all[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct
    #ret["forget_prompts_correct"] = forget_correct#[not item for item in forget_probs]

    return ret

def get_orignal_response(edit_model,model, tok, prompts: typing.List[str]):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        correct_id = torch.argmax(gathered, dim=1)

    with torch.no_grad():
        logits = edit_model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)
    
    return (ans == correct_id).detach().cpu().numpy().tolist()


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        if 'llama' in model.config._name_or_path.lower():
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()
