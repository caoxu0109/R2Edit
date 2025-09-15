import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
import sys
from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre/all.json"


class RelationEditDataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "all.json"
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, chucks in enumerate(raw):
            if len(chucks)-1 == 1:
                step1 = chucks['original']
                assert (
                "nq question: " in step1["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
                step1_toks = tok(" " + step1["loc_ans"])["input_ids"]
                data.append(
                    [{
                        "case_id": i,
                        "requested_rewrite": {
                            "prompt": step1["src"].replace(step1["subject"], "{}"),
                            "subject": step1["subject"],
                            "target_new": {"str": step1["answers"][0]},
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
                    }])
            elif len(chucks)-1 == 2:
            # This is the second step, which has two chunks.
                step1 = chucks['original']
                step2 = chucks['target']
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
                        "target_new": {"str": step1["answers"][0]},
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
                        "target_new": {"str": step2["alt"]},
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
                ])
            elif len(chucks)-1 == 3:
                step1 = chucks['original']
                step2 = chucks['target']
                step3 = chucks['third_step']
                assert (
                    "nq question: " in step1["loc"]
                ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
                step1_toks = tok(" " + step1["loc_ans"])["input_ids"]
                step2_toks = tok(" " + step2["loc_ans"])["input_ids"]
                step3_toks = tok(" " + step3["loc_ans"])["input_ids"]
                data.append(
                    [{
                        "case_id": i,
                        "requested_rewrite": {
                            "prompt": step1["src"].replace(step1["subject"], "{}"),
                            "subject": step1["subject"],
                            "target_new": {"str": step1["answers"][0]},
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
                            "target_new": {"str": step2["alt"]},
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
                    },
                    {
                        "case_id": i,
                        "requested_rewrite": {
                            "prompt": step3["src"].replace(step3["subject"], "{}"),
                            "subject": step3["subject"],
                            "target_new": {"str": step3["alt"]},
                            "target_true": {"str": "<|endoftext|>"},
                        },
                        "paraphrase_prompts": [step3["rephrase"]],
                        "neighborhood_prompts": [
                            {
                                "prompt": step3["loc"] + "?" + tok.decode(step3_toks[:i]),
                                "target": tok.decode(step3_toks[i]),
                            }
                            for i in range(len(step3_toks))
                        ],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                    ]
                )
            else:
                raise ValueError(f"Unexpected number of chunks: {len(chucks)}. Expected 1, 2, or 3.")

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

