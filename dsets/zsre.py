import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        #zsre_loc = data_dir / "zsre_mend_eval.json"
        zsre_loc = Path("./data/llama3-8b-it/zsre/puz.json")
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            #record = chunk['step1']
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["alt"]},
                        "target_true": {"str": "<|endoftext|>"},
                        "target_old": {"str": record["answers"][0]},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attack_probes_p":{"str": record["attack_probes_p"]},
                    "attack_probes_em":{"str": record["attack_probes_em"]},
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
