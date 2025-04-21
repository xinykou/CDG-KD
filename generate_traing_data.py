import os
import sys

# sys.path.append("/media/5/yx/distill_watermark")
# sys.path.append("/media/5/yx/distill_watermark/watermark/distill/utils")
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

print(sys.path)
import argparse
from watermark.utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark


def generate_batch(
    model, tokenizer, instruction, batch_size=10, watermark=None, watermark_name=None
):

    org_instructions = [instruction] * batch_size

    instructions = []
    for query in org_instructions:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        instructions.append(inputs)

    inputs = tokenizer(instructions, return_tensors="pt").to(model.device)
    if (
        ("KGW" in watermark_name)
        or ("SynthID" in watermark_name)
        or ("Unigram" in watermark_name)
    ):
        outputs = watermark.batch_generate_watermarked_tokens(inputs)
    elif "DIP" in watermark_name:
        outputs = watermark.generate_watermarked_text(instructions[0])
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        print("!!! No watermark !!!")

    responses = (
        tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if "DIP" not in watermark_name
        else outputs
    )

    filtered_responses = []

    for response in responses:
        parts = response.split("[[")
        parts = [r.split("]]")[0] for r in parts]

        for part in parts:
            if "Instruction:" in part and "Input:" in part and "Answer:" in part:
                instruction = part.split("Instruction:")[1].split("Input:")[0].strip()
                input = part.split("Input:")[1].split("Answer:")[0].strip()
                answer = part.split("Answer:")[1].strip()
                filtered_responses.append(
                    {"Instruction": instruction, "Input": input, "Answer": answer}
                )

    return filtered_responses


def worker(
    rank,
    t,
    instruction,
    model_path,
    output_dir,
    samples_per_worker,
    batch_size,
    watermark_type,
    config_file,
):
    # num_gpus = torch.cuda.device_count()
    device_id = rank

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .to(f"cuda:{device_id}")
        .eval()
    )

    myWatermark = None
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device=model.device,
        max_new_tokens=2000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    # if "DIP" in watermark_type:
    #     transformers_config.min_new_tokens = 2000

    if "KGW" in watermark_type:
        myWatermark = AutoWatermark.load(
            "KGW",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
        print(f"watermark name---->{watermark_type}")
    elif "SynthID" in watermark_type:
        myWatermark = AutoWatermark.load(
            "SynthID",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
        print(f"watermark name---->{watermark_type}")
    elif "Unigram" in watermark_type:
        transformers_config.vocab_size = 151552  # glm-4-9b-chat
        myWatermark = AutoWatermark.load(
            "Unigram",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
        print(f"watermark name---->{watermark_type}")
    elif "DIP" in watermark_type:
        myWatermark = AutoWatermark.load(
            "DIP",
            algorithm_config=config_file,
            transformers_config=transformers_config,
        )
        print(f"watermark name---->{watermark_type}")
    else:
        print(f"!!! No watermark !!!")

    output_file = os.path.join(output_dir, f"worker_{rank}_{t}.json")

    pbar = tqdm(total=samples_per_worker, position=rank, desc=f"Worker {rank}")
    generated_samples = 0
    with open(output_file, "w", encoding="utf-8") as file:
        while generated_samples < samples_per_worker:
            batch = generate_batch(
                model,
                tokenizer,
                instruction,
                batch_size,
                watermark=myWatermark,
                watermark_name=watermark_type,
            )
            for response in batch:
                file.write(json.dumps(response, ensure_ascii=False))
                file.write("\n")
                file.flush()
                generated_samples += 1
                pbar.update(1)
                if generated_samples >= samples_per_worker:
                    break

    pbar.close()
    return output_file


def merge_files(output_files, final_output):
    with open(final_output, "w", encoding="utf-8") as outfile:
        for filename in output_files:
            with open(filename, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/media/5/yx/model_cache/glm-4-9b-chat"
    )
    parser.add_argument(
        "--total_samples", type=int, default=80, help="(batch_size * 20) each time"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="data/distillation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--final_output", type=str, default="dip_prefix_1.json")
    parser.add_argument("--watermark_name", type=str, default="")
    parser.add_argument(
        "--config_file", type=str, default="src/distill/config/DIP_1.json"
    )
    args = parser.parse_args()

    with open("watermark/distill/instruction.txt", "r", encoding="utf-8") as file:
        instruction = file.read().strip()

    os.makedirs(args.output_dir, exist_ok=True)
    final_output = os.path.join(args.output_dir, args.final_output)

    mp.set_start_method("spawn", force=True)

    num_workers = torch.cuda.device_count()
    samples_per_worker = args.total_samples // num_workers

    t = int(time.time())

    with Pool(processes=num_workers) as pool:
        output_files = pool.starmap(
            worker,
            [
                (
                    i,
                    t,
                    instruction,
                    args.model_path,
                    args.output_dir,
                    samples_per_worker,
                    args.batch_size,
                    args.watermark_name,
                    args.config_file,
                )
                for i in range(num_workers)
            ],
        )

    merge_files(output_files, final_output)

    # Clean up temporary files
    for file in output_files:
        os.remove(file)

    print(f"Generated {args.total_samples} samples in {final_output}")
