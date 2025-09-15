import os
from pathlib import Path
import torch
import apache_beam as beam
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="/data/jianghc/llama3-8b-instruct", choices=["gpt2-xl", "EleutherAI/gpt-j-6B","/data/jianghc/llama3-8b-instruct"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[4,5,6,7,8], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        # proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        # layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"
        layer_name = f"model.layers.{layer_num}.mlp.down_proj"
        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('data/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en",)[ds_name],
            cache_dir = '/mnt/data/wu/caoxu/relation_edit/AlphaEdit-main/catch',
            beam_runner='DirectRunner',          # 二选一
           # beam_options=beam.PipelineOptions()
        )
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 1  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


def compute_new_knowledge_stats(
    knowledge_list,  # 新增参数：包含100条知识的列表
    model,
    tokenizer,
    layer_name,
    to_collect=["mom2"],
    sample_size=None,
    precision=None,
    batch_tokens=None,
    force_recompute=False,
    progress=tqdm
):
    """
    计算新增知识列表的特征矩阵
    
    Args:
        knowledge_list: 包含新增知识的列表，长度为100
        model: 要分析的模型
        tokenizer: 对应的tokenizer
        layer_name: 要分析的层名称
        to_collect: 要收集的统计量类型
        sample_size: 样本大小
        precision: 计算精度
        batch_tokens: 批次token数量
        force_recompute: 是否强制重新计算
        progress: 进度条函数
    
    Returns:
        stat: 计算得到的统计结果
    """
    
    def get_ds():
        # 确定模型的最大序列长度
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        # 从结构化数据中提取文本内容，并转换为TokenizedDataset期望的格式
        text_data = []
        for item in knowledge_list:
            if isinstance(item, list):
                # 如果item是列表，遍历其中的字典
                for sub_item in item:
                    if isinstance(sub_item, dict) and "requested_rewrite" in sub_item:
                        # 提取prompt和subject组合
                        rewrite = sub_item["requested_rewrite"]
                        prompt = rewrite["prompt"].format(rewrite["subject"])
                        text_data.append({"text": prompt})
                        
                        # 也可以添加paraphrase_prompts
                        if "paraphrase_prompts" in sub_item:
                            for para in sub_item["paraphrase_prompts"]:
                                text_data.append({"text": para})
                        
                        # 添加neighborhood_prompts
                        if "neighborhood_prompts" in sub_item:
                            for neighbor in sub_item["neighborhood_prompts"]:
                                if "prompt" in neighbor:
                                    text_data.append({"text": neighbor["prompt"]})
            elif isinstance(item, dict) and "requested_rewrite" in item:
                # 如果item直接是字典
                rewrite = item["requested_rewrite"]
                prompt = rewrite["prompt"].format(rewrite["subject"])
                text_data.append({"text": prompt})
                
                if "paraphrase_prompts" in item:
                    for para in item["paraphrase_prompts"]:
                        text_data.append({"text": para})
                
                if "neighborhood_prompts" in item:
                    for neighbor in item["neighborhood_prompts"]:
                        if "prompt" in neighbor:
                            text_data.append({"text": neighbor["prompt"]})
        
        # 使用提取的文本数据创建TokenizedDataset
        return TokenizedDataset(text_data, tokenizer, maxlen=maxlen)

    # 获取模型配置参数
    batch_size = 1  # 每次检查的数据集文本数量
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    
    print(f"Computing Cov for new knowledge locally....")

    # 创建数据集
    ds = get_ds()
    if progress is None:
        progress = lambda x: x

    # 初始化统计对象
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    
    # 创建数据加载器
    loader = tally(
        stat,
        ds,
        cache=None,  # 不使用缓存
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    
    # 计算批次数量
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    
    # 执行特征提取和统计计算
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in loader:
                batch = dict_to_(batch[0], "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
        
    return stat

# if __name__ == "__main__":

#     compute_new_knowledge_stats()
