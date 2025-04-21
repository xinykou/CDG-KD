import os
import sys

print(sys.path)
import argparse
import json
import torch
from watermark.utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# from datasets import load_dataset

# data = load_dataset("m-a-p/SuperGPQA", cache_dir="./cache")
# print("finished")


def convert_tensor_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_native(element) for element in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    else:
        return obj


def main(args):
    data_path = args.data_path
    watermark_tokenizer_name = args.watermark_tokenizer_name

    if data_path.endswith("jsonl"):
        with open(data_path, "r") as file:
            lines = file.readlines()
    elif data_path.endswith("json"):
        with open(data_path, "r") as file:
            lines = json.load(file)
    else:
        raise ValueError("dataset invalid !!!")

    ### tokenizer must be from source model, chatglm-4-9b
    tokenizer = AutoTokenizer.from_pretrained(
        watermark_tokenizer_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        watermark_tokenizer_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    device = model.device
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device=device,
    )

    watermark_type = args.watermark_name
    config_file = args.watermark_config
    myWatermark = None
    if "KGW" in watermark_type:
        myWatermark = AutoWatermark.load(
            "KGW",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
    elif "SynthID" in watermark_type:
        myWatermark = AutoWatermark.load(
            "SynthID",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
    elif "Unigram" in watermark_type:
        transformers_config.vocab_size = 151552  # glm-4-9b-chat
        myWatermark = AutoWatermark.load(
            "Unigram",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
    elif "Unbiased" in watermark_type:
        myWatermark = AutoWatermark.load(
            "Unbiased",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
    elif "DIP" in watermark_type:
        myWatermark = AutoWatermark.load(
            "DIP", algorithm_config=config_file, transformers_config=transformers_config
        )
    else:
        raise NameError(f"Watermark name ---> {watermark_type} is not supported")

    batch_size = args.batch_size
    zscore_group = []
    pvalue_group = []
    all_log10_p_values = []
    error_nums = 0
    if args.max_samples != -1:
        lines = lines[: args.max_samples]
    for index in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
        if data_path.endswith("jsonl"):
            block_line = []
            for line in lines[index : index + batch_size]:
                data = json.loads(line)
                block_line.append(data["predict"])
        else:
            block_line = [li["output"] for li in lines[index : index + batch_size]]

        for text in block_line:
            try:
                res = myWatermark.detect_watermark(text)
            except Exception as e:
                print(f"检测水印时发生错误：{e}")
                error_nums += 1
                continue
            zscore_group.append(res["score"])
            pvalue_group.append(res["p_value"])
            log10_p_values = -np.log10(res["p_value"])
            all_log10_p_values.append(log10_p_values)

    valid_nums = len(zscore_group)
    zscore_group.sort()
    pvalue_group.sort()

    # # Calculate and print final results
    average_log10_p_value = sum(all_log10_p_values) / len(all_log10_p_values)
    print(f"average log10(p-value): {average_log10_p_value}")

    median_z_score = zscore_group[len(zscore_group) // 2]
    median_p_value = pvalue_group[len(pvalue_group) // 2]

    print(f"watermak median z_score: {median_z_score}")
    print(f"watermak median p_value: {median_p_value}")
    print(f"error nums/valid nums: {error_nums}/{valid_nums}")

    # 构造要写入的 JSON 对象
    data_to_write = {
        "watermark_median_z_score": median_z_score,
        "watermark_median_p_value": median_p_value,
        "error_nums_valid_nums_ratio": f"{error_nums}/{valid_nums}",
    }

    # # 打开文件并写入数据
    if args.is_training_data or args.not_record_res:
        pass
    else:
        with open(data_path, "a") as file:  # 使用 "a" 模式追加数据
            file.write(
                json.dumps(convert_tensor_to_native(data_to_write)) + "\n"
            )  # 写入一行 JSON 数据并换行

    print("dataset from {} has been processed".format(data_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark_tokenizer_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--watermark_name",
        type=str,
        default=None,
        choices=[
            "KGW_1",
            "KGW_2",
            "KGW_3",
            "No_watermark",
            "SynthID_1",
            "SynthID_2",
            "SynthID_3",
            "Unigram",
            "Unbiased_1",
            "Unbiased_2",
            "Unbiased_3",
            "DIP_1",
            "DIP_2",
            "DIP_3",
        ],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="The maximum number of samples to evaluate",
    )
    parser.add_argument("--watermark_config", type=str, default=None)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--is_training_data", action="store_true")
    parser.add_argument(
        "--not_record_res",
        action="store_true",
        help="whether to record the result in the dataset",
    )
    args = parser.parse_args()
    main(args)
