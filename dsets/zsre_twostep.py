import json
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer
import sys
from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/relation_edit/train_relIDK.json"


class RelaTwoStepDataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = Path("./data/relation_edit/all_rel_IDK.json")
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, chucks in enumerate(raw):
            step1 = chucks['step1']
            step2 = chucks['step2']
            assert (
                "nq question: " in step1["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            step1_toks = tok(" " + step1["loc_ans"])["input_ids"]
            step2_toks = tok(" " + step2["loc_ans"])["input_ids"]
            data.append(
                [{
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": step1["src"].replace(step1["subject"], "{}"),
                        "subject": step1["subject"],
                        "target_orignal": {"str": step1["alt"]},
                        "target_new": {"str": step1["answers"][0]},
                        "target_new2": {"str": "I don't Know"},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [step1["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": step1["loc"] + "?" + tok.decode(step1_toks[:i]),
                            "target": tok.decode(step1_toks[i]),
                        }
                        for i in range(len(step1_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                },
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": step2["src"].replace(step2["subject"], "{}"),
                        "subject": step2["subject"],
                        "target_orignal": {"str": step2["alt"]},
                        "target_new": {"str": step2["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [step2["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": step2["loc"] + "?" + tok.decode(step2_toks[:i]),
                            "target": tok.decode(step2_toks[i]),
                        }
                        for i in range(len(step2_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
                ]
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
