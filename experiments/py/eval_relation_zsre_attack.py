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
from typing import Literal, Tuple, List, Dict, Union
from dsets import AttributeSnippets



def generate_target_tokens_argmax(
    model: AutoModelForCausalLM, tok: AutoTokenizer, prefix: str, n_steps: int = 1
):
    generated_tokens = {
        "id": [],
        "str": "",
    }
    for i in range(n_steps):
        input_toks = tok(prefix, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**input_toks).logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = probs.argmax(dim=-1).item()
        next_str = tok.decode(next_id)
        prefix += next_str
        generated_tokens["id"].append(next_id)
        generated_tokens["str"] += next_str
    return generated_tokens


def get_target_probability_2(model, tok, prefixes: Union[List[str], str], targets: Union[List[str], str]):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if isinstance(targets, str):
        targets = [targets]

    assert len(prefixes) == len(targets)
    new_targets = []
    for tgt in targets:
        new_targets.append(tgt if tgt[0] == " " else " " + tgt)
    targets = new_targets
    target_ids = [tok(tgt, return_tensors="pt")["input_ids"][0] for tgt in targets]

    prefix_lens = [len(tok.encode(x)) for x in prefixes]
    sentence_toks = tok(
        [p + t for p, t in zip(prefixes, targets)], return_tensors="pt", padding=True
    ).to("cuda")
    with torch.no_grad():
        logits = model(**sentence_toks).logits
    probs = torch.softmax(logits, dim=-1)
    ret = []
    for i in range(len(prefixes)):
        ps = []
        for j, cur_tok in enumerate(target_ids[i]):
            ps.append(probs[i, prefix_lens[i] - 1 + j, cur_tok].item())
        p = np.prod(ps).item()
        ret.append(p)
    return ret



def evaluate_superficial_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    probe_key: str = "attack_probes_p",
):
    if "requested_rewrite" in record:
        target_true_ids = tok.encode(
            f" {record['requested_rewrite']['target_old']['str']}"
        )
        target_new_ids = tok.encode(
            f" {record['requested_rewrite']['target_new']['str']}"
        )
        target_true_str = record["requested_rewrite"]["target_old"]["str"]
        target_new_str = record["requested_rewrite"]["target_new"]["str"]
    else:
        target_true_ids = tok.encode(f" {record['alt']}")
        target_new_ids = tok.encode(f" {record['answers'][0]}")
        target_true_str = record["alt"]
        target_new_str = record["answers"][0]
    test_prompts = record[probe_key]['str']

    original_em = []
    new_em = []
    p_olds, p_news = [], []
    for probe in test_prompts:
        prediction = generate_target_tokens_argmax(
            model, tok, probe, n_steps=len(target_true_ids)
        )
        original_cond = (
            prediction["id"] == target_true_ids
            or prediction["str"].strip() == target_true_str
        )
        original_em.append(original_cond)

        prediction = generate_target_tokens_argmax(
            model, tok, probe, n_steps=len(target_new_ids)
        )
        new_cond = (
            prediction["id"] == target_new_ids
            or prediction["str"].strip() == target_new_str
        )
        new_em.append(new_cond)

    p_olds = get_target_probability_2(
        model, tok, test_prompts, [target_true_str for _ in test_prompts]
    )
    p_news = get_target_probability_2(
        model, tok, test_prompts, [target_new_str for _ in test_prompts]
    )
    old_gt_new = [po > pn for po, pn in zip(p_olds, p_news)]
    return original_em, new_em, p_olds, p_news, old_gt_new

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
    #forget_prompts = record["forget_prompts"]
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
    # evaluate_superficial_editing(model=model, tok=tok, record=record, probe_key="attack_probes_p")
    original_correct, new_correct, old_probs, new_probs, old_gt_new = evaluate_superficial_editing(model=model, tok=tok, record=record, probe_key="attack_probes_em")
    #ret["forget_prompts_correct"] = forget_correct#[not item for item in forget_probs]
    ret["attack_probes_p_original_correct"] = original_correct
    ret["attack_probes_p_new_correct"] = new_correct
    ret["attack_probes_p_old_probs"] = old_probs
    ret["attack_probes_p_new_probs"] = new_probs
    ret["attack_probes_p_old_gt_new"] = old_gt_new

    original_correct_em, new_correct_em, old_probs_em, new_probs_em, old_gt_new_em = evaluate_superficial_editing(model=model, tok=tok, record=record, probe_key="attack_probes_em")
    ret["attack_probes_em_original_correct"] = original_correct_em
    ret["attack_probes_em_new_correct"] = new_correct_em
    ret["attack_probes_em_old_probs"] = old_probs_em
    ret["attack_probes_em_new_probs"] = new_probs_em
    ret["attack_probes_em_old_gt_new"] = old_gt_new_em


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
