from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import time
import random
import torch.multiprocessing as mp


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch):
    return batch


def process_batch(
    process_id: int, time_stamp: int, prompts: List[str], inputs_list: List, args
):
    # Device
    device = f"cuda:{process_id}" if torch.cuda.is_available() else "cpu"

    # Create temporary output file
    temp_output = (
        Path(args.output_file).parent / f"temp_{process_id}_{time_stamp}.jsonl"
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader for this process's portion
    dataset = TextDataset(prompts)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False
    )

    if args.watermark_type != "No_watermark":
        myWatermark = None
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            device=model.device,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            min_length=30 + args.max_new_tokens,
        )

        if "KGW" in args.watermark_type:
            myWatermark = AutoWatermark.load(
                "KGW",
                algorithm_config=args.config_file,
                transformers_config=transformers_config,
            )
        elif "SynthID" in args.watermark_type:
            myWatermark = AutoWatermark.load(
                "SynthID",
                algorithm_config=args.config_file,
                transformers_config=transformers_config,
            )

        elif "ReverseWatermark" == args.watermark_type:
            myWatermark = AutoWatermark.load(
                "ReverseWatermark",
                algorithm_config=args.config_file,
                transformers_config=transformers_config,
                attack_type=args.attack_type,
            )
        elif "Unigram" in args.watermark_type:
            transformers_config.vocab_size = 151552
            myWatermark = AutoWatermark.load(
                "Unigram",
                algorithm_config=args.config_file,
                transformers_config=transformers_config,
            )
        elif "DIP" in args.watermark_type:
            myWatermark = AutoWatermark.load(
                "DIP",
                algorithm_config=args.config_file,
                transformers_config=transformers_config,
            )

        print(f"Watermark Name------> {args.watermark_type}")
    else:
        print(f"Watermark Name------> No watermark")

    with open(temp_output, mode="w") as f:
        # Set up progress bar for this process
        pbar = tqdm(
            total=len(dataloader),
            desc=f"Process {process_id} Generating",
            position=process_id,
        )

        for index, batch_prompts in enumerate(dataloader):
            batch_inputs = inputs_list[
                index * args.batch_size : (index + 1) * args.batch_size
            ]
            encoded_prompt = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(device)
            if (
                args.watermark_type != "No_watermark"
                and "DIP" not in args.watermark_type
            ):
                generated_texts = myWatermark.batch_generate_watermarked_tokens(
                    encoded_prompt
                )
            elif "DIP" in args.watermark_type:
                generated_texts = myWatermark.generate_watermarked_text(
                    batch_prompts[0]
                )
                print(f"watermark type: {args.watermark_type}")

            else:
                generated_texts = model.generate(
                    **encoded_prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_length=30 + args.max_new_tokens,
                    do_sample=True,
                )
                # truncate prompt
                generated_texts = generated_texts[
                    :, encoded_prompt["input_ids"].shape[1] :
                ]
            if "DIP" not in args.watermark_type:
                decoded_texts = tokenizer.batch_decode(
                    generated_texts, skip_special_tokens=True
                )
            else:
                decoded_texts = [generated_texts]

            for idx in range(len(batch_prompts)):
                if args.is_training_data or args.is_test_safety:
                    f.write(
                        json.dumps(
                            {
                                "input": (
                                    batch_inputs[idx]["input"]
                                    if "input" in batch_inputs[idx]
                                    else ""
                                ),
                                "instruction": batch_inputs[idx]["instruction"],
                                "output": decoded_texts[idx],
                            }
                        )
                        + "\n"
                    )

                else:
                    f.write(
                        json.dumps(
                            {
                                "prompt": batch_prompts[idx],
                                "predict": decoded_texts[idx],
                            }
                        )
                        + "\n"
                    )
                f.flush()

            pbar.update(1)

        pbar.close()

    return temp_output


def main(args):
    # Load data
    if args.data_path.endswith(
        "json"
    ):  # for "c4" data or "dolly-writing-prompts" data or "HarmfulQ_AdvBench" data
        with open(args.data_path, "r") as f:  # json data is loaded
            data = json.load(f)

    elif args.data_path.endswith("txt"):  # for "MaliciousInstruct" data
        with open(args.data_path, "r") as f:
            org_data = f.readlines()
            data = []
            for d in org_data:
                data.append({"instruction": d.strip()})

    inputs_list = []
    if "c4" in args.data_path:
        org_data = data
        data = []
        for d in org_data:
            data.append({"instruction": d["text"]})
    if (
        "HarmfulQ_AdvBench" in args.data_path
        or "MaliciousInstruct" in args.data_path
        or "dolly" in args.data_path
        or "c4" in args.data_path
    ):
        if args.template == "default":
            prompts = [
                (
                    "Human: " + d["instruction"] + "\n" + d["input"] + "\nAssistant:"
                    if "input" in d
                    else "Human: " + d["instruction"] + "\nAssistant:"
                )
                for d in data
            ]
        elif args.template == "self":
            print(f"model path: {args.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, trust_remote_code=True
            )
            prompts = []
            for d in data:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": d["instruction"]}], tokenize=False
                )
                prompts.append(prompt)
        else:
            prompts = [d["instruction"] for d in data]
            # raise ValueError("Dataset name is not supported !")

    elif args.is_training_data:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        org_prompts = [d["instruction"] + "\n" + d["input"] for d in data]
        prompts = []
        for d in org_prompts:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": d}], tokenize=False
            )
            prompts.append(prompt)

    ### data for recover
    if args.is_test_safety or args.is_training_data:
        for item in data:
            inputs_list.append(
                {
                    "input": item["input"] if "input" in item else "",
                    "instruction": item["instruction"],
                }
            )

    if args.start_idx is not None and args.end_idx is not None:
        prompts = prompts[args.start_idx : args.end_idx]

    elif args.max_samples > 0:
        prompts = prompts[: args.max_samples]

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of processes based on available GPUs
    num_processes = min(args.num_processes, torch.cuda.device_count())

    # Split prompts among processes
    prompts_per_process = len(prompts) // num_processes
    prompt_splits = [
        prompts[i : i + prompts_per_process]
        for i in range(0, len(prompts), prompts_per_process)
    ]

    if args.is_training_data or args.is_test_safety:
        inputs_list_splits = [
            inputs_list[i : i + prompts_per_process]
            for i in range(0, len(inputs_list), prompts_per_process)
        ]

    time_stamp = int(time.time())

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            future = executor.submit(
                process_batch,
                i,
                time_stamp,
                prompt_splits[i],
                (
                    inputs_list_splits[i]
                    if args.is_training_data or args.is_test_safety
                    else []
                ),
                args,
            )
            futures.append(future)

        # Wait for all processes to complete and get temp file paths
        temp_files = [future.result() for future in futures]

    # Merge all temporary files
    if args.is_training_data or args.is_test_safety:
        all_data = []  # 存储所有 JSON 对象的列表
        for temp_file in temp_files:
            with open(temp_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    all_data.append(json.loads(line.strip()))  # 解析 JSON 行并加入列表
            time.sleep(1)
            os.remove(temp_file)  # 删除临时文件

        # 写入最终 JSON 文件
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(
                all_data, outfile, indent=4, ensure_ascii=False
            )  # 保存为 JSON 数组

    else:
        with open(output_file, "w") as outfile:
            for temp_file in temp_files:
                with open(temp_file, "r") as infile:
                    outfile.write(infile.read())
                # Remove temporary file
                time.sleep(1)
                os.remove(temp_file)

    print(f"\nGeneration completed. Results saved to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Parallel text generation with watermark"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/panleyi/LLaMA-Factory/data/c4_truncate.json",
        help="Path to input data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/panleyi/LLaMA-Factory/saves/Llama-7b/full/sft/wm_glm",
        help="Path to the model",
    )
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process. -1 for all",
    )
    parser.add_argument("--start_idx", type=int, default=None, help="Start index")
    parser.add_argument("--end_idx", type=int, default=None, help="End index")
    parser.add_argument(
        "--watermark_type", type=str, default=None, help="Type of watermark to add"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/KGW.json",
        help="Path to watermark configuration file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for parallel processing"
    )
    parser.add_argument(
        "--num_processes", type=int, default=2, help="Number of precesses"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=600,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save output files",
    )
    parser.add_argument(
        "--is_training_data",
        action="store_true",
        help="Set this flag if the data is for distillation training.",
    )
    parser.add_argument(
        "--is_test_safety",
        action="store_true",
        help="Set this flag if the data is for safety evaluation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="",
        choices=["scrubbing_attack", "spoofing_attack"],
    )
    args = parser.parse_args()
    main(args)
