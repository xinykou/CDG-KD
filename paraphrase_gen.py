import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.tokenize import sent_tokenize
import time
import argparse
import json
from pathlib import Path
import math
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import Pool
from parrot import Parrot

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27888"
os.environ["HTTP_PROXY"] = "http://127.0.0.1.27888"

import nltk
from nltk.tokenize import sent_tokenize

download_dir = "/media/5/yx/nltk_data"
nltk.download("punkt", download_dir=download_dir)
if download_dir not in nltk.data.path:
    nltk.data.path.append(download_dir)


# text = "Hello! This is a test. Let's see if nltk sentence tokenization works properly."

# sentences = sent_tokenize(text)

# print(sentences)

# exit()


class DipperParaphraser(object):
    def __init__(
        self,
        model="/media/5/yx/model_cache/dipper-paraphraser-xxl",
        verbose=True,
        div=20,
        device=None,
    ):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, device_map=device, torch_dtype=torch.bfloat16
        )
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.eval()
        self.lex_diversity = div
        self.order_diversity = div
        self.device = device

    def paraphrase(self, input_text, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert self.lex_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert self.order_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        kwargs["do_sample"] = True
        kwargs["top_p"] = 0.75
        kwargs["top_k"] = None
        kwargs["max_length"] = 256

        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt").to(
                self.device
            )
            final_input = {k: v for k, v in final_input.items()}

            with torch.inference_mode():
                try:
                    outputs = self.model.generate(**final_input, **kwargs)
                except:
                    print("!!!!!!!! Error in generating !!!!!")
                    return None
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text

    def batch_paraphrase(self, input_texts, prefixs=None, sent_interval=3, **kwargs):
        texts = []
        if prefixs is None:
            prefixs = ["" for _ in range(len(input_texts))]
        for input_text, prefix in zip(input_texts, prefixs):
            out = self.paraphrase(input_text, prefix, sent_interval, **kwargs)
            texts.append(out)
        return texts


class PegasusParaphraser:
    def __init__(self, temp=1.5, device=None):
        model_name = "/media/5/yx/model_cache/pegasus_paraphrase"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.temp = temp
        self.device = device

    def paraphrase(self, input_text, prefix=""):
        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)

        batch = self.tokenizer(
            sentences,
            max_length=60,
            truncation=True,
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        try:
            para_toks = self.model.generate(
                **batch,
                do_sample=True,
                max_length=60,
                num_return_sequences=1,
                temperature=self.temp,
            )
        except:
            print("!!!!!!!! Error in generating !!!!!")
            return None
        out_text = " " if input_text.startswith(" ") else ""
        for one_tok in para_toks:
            new_text = self.tokenizer.decode(one_tok, skip_special_tokens=True)
            out_text = out_text + " " + new_text

        return out_text

    def batch_paraphrase(self, input_texts):
        all_sent = []
        st_id = []
        ed_id = []
        for input_text in input_texts:
            st_id.append(len(all_sent))
            input_text = " ".join(input_text.split())
            sentences = sent_tokenize(input_text)
            all_sent.extend(sentences)
            ed_id.append(len(all_sent))
        batch = self.tokenizer(
            all_sent,
            max_length=60,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")
        para_toks = []
        for i in range(0, len(batch["input_ids"]), bsize):
            cur_toks = self.model.generate(
                input_ids=batch["input_ids"][i : i + bsize],
                attention_mask=batch["attention_mask"][i : i + bsize],
                do_sample=True,
                max_length=60,
                num_return_sequences=1,
                temperature=self.temp,
            )
            para_toks.append(cur_toks)
        padded_toks = []
        maxlen = max([tok.shape[1] for tok in para_toks])
        for tok in para_toks:
            new_tok = torch.nn.functional.pad(
                tok, (0, maxlen - tok.shape[1]), value=self.tokenizer.pad_token_id
            )
            padded_toks.append(new_tok)
        para_toks = torch.cat(padded_toks, dim=0)
        all_decoded = self.tokenizer.batch_decode(para_toks, skip_special_tokens=True)
        out_texts = []
        for i in range(len(input_texts)):
            out_text = " " if input_texts[i].startswith(" ") else ""
            for new_text in all_decoded[st_id[i] : ed_id[i]]:
                out_text = out_text + " " + new_text
            out_texts.append(out_text)
        return out_texts


class SParrotParaphraser(Parrot):
    def __init__(
        self,
        model_tag="/media/5/yx/model_cache/parrot_paraphraser_on_T5",
        device=None,
        use_gpu=True,
    ):
        super().__init__(model_tag, use_gpu)
        self.device = device

    def paraphrase(
        self,
        input_phrase,
        diversity_ranker="levenshtein",
        do_diverse=False,
        max_return_phrases=1,
        max_length=256,
        adequacy_threshold=0.8,
        fluency_threshold=0.8,
    ):

        self.model = self.model.to(self.device)

        import re

        save_phrase = input_phrase
        if len(input_phrase) >= max_length:
            max_length += 32

        input_phrase = re.sub("[^a-zA-Z0-9 \?'\-\/\:\.]", "", input_phrase)
        input_phrase = "paraphrase: " + input_phrase
        input_ids = self.tokenizer.encode(input_phrase, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        if do_diverse:
            for n in range(2, 9):
                if max_return_phrases % n == 0:
                    break
            # print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)
            preds = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=256,
                num_beams=max_return_phrases,
                num_beam_groups=n,
                diversity_penalty=2.0,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )
        else:
            preds = self.model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=256,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
            gen_pp = re.sub("[^a-zA-Z0-9 \?'\-]", "", gen_pp)
            paraphrases.add(gen_pp)

        adequacy_filtered_phrases = self.adequacy_score.filter(
            input_phrase, paraphrases, adequacy_threshold, self.device
        )
        if len(adequacy_filtered_phrases) == 0:
            adequacy_filtered_phrases = paraphrases
        fluency_filtered_phrases = self.fluency_score.filter(
            adequacy_filtered_phrases, fluency_threshold, self.device
        )
        if len(fluency_filtered_phrases) == 0:
            fluency_filtered_phrases = adequacy_filtered_phrases
        diversity_scored_phrases = self.diversity_score.rank(
            input_phrase, fluency_filtered_phrases, diversity_ranker
        )
        para_phrases = []
        for para_phrase, diversity_score in diversity_scored_phrases.items():
            para_phrases.append((para_phrase, diversity_score))
        para_phrases.sort(key=lambda x: x[1], reverse=True)
        para_phrases = [x[0] for x in para_phrases]
        return para_phrases[0]  # return the first one


def process_paraphraser(process_id, data, args):
    print(f"Process {process_id} started: {len(data)} samples")
    if args.paraphraser == "dipper":
        device = f"cuda:{process_id}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0"
    if args.paraphraser == "dipper":
        paraphraser = DipperParaphraser(device=device)
    elif args.paraphraser == "pegasus":
        paraphraser = PegasusParaphraser(device=device)
    elif args.paraphraser == "parrot":
        paraphraser = SParrotParaphraser(device=device)
    else:
        raise NotImplementedError("Unknown paraphraser: %s" % args.paraphraser)

    temp_output = Path(args.output_file).parent / f"temp_{process_id}.json"

    with open(temp_output, "w") as f:
        batch_size = args.batch_size
        if batch_size == 1:
            pbar = tqdm(
                total=len(data),
                desc=f"Process {process_id} Generating",
                position=process_id,
            )
        else:
            pbar = tqdm(
                total=math.ceil(len(data) / batch_size),
                desc=f"Process {process_id} Generating",
                position=process_id,
            )
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            instruction = [batch[index]["instruction"] for index in range(len(batch))]
            input = [batch[index]["input"] for index in range(len(batch))]
            answer = [batch[index]["output"] for index in range(len(batch))]
            if batch_size == 1:
                instruction = instruction[0]
                input = input[0]
                answer = answer[0]
                paraphrased_answer = paraphraser.paraphrase(answer)

            else:
                paraphrased_answer = paraphraser.batch_paraphrase(answer)

            if batch_size == 1:
                f.write(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "input": input,
                            "output": (
                                paraphrased_answer
                                if paraphrased_answer is not None
                                else answer
                            ),
                        }
                    )
                    + "\n"
                )
            else:
                for index in range(len(batch)):
                    f.write(
                        json.dumps(
                            {
                                "instruction": instruction[index],
                                "input": input[index],
                                "output": (
                                    paraphrased_answer[index]
                                    if paraphrased_answer[index] is not None
                                    else answer[index]
                                ),
                            }
                        )
                        + "\n"
                    )
            f.flush()
            pbar.update(1)
        pbar.close()

    return temp_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paraphraser",
        type=str,
        default="pegasus",
        help="The paraphraser to use.",
        choices=["dipper", "pegasus", "parrot"],
    )
    parser.add_argument(
        "--num_workers", default=2, type=int, help="Number of workers to use."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/direct_distillation/kgw_prefix_1_final.json",
        help="The data path.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/paraphrase/kgw_prefix_1_final_paraphrased.json",
        help="The output file.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.max_samples != -1:
        data = data[: args.max_samples]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.paraphraser == "dipper":
        num_processes = min(args.num_workers, torch.cuda.device_count())
        responses_per_process = len(data) // num_processes
    else:
        num_processes = args.num_workers
        responses_per_process = len(data) // num_processes

    responses_splits = [
        data[i * responses_per_process : (i + 1) * responses_per_process]
        for i in range(num_processes)
    ]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        features = [
            executor.submit(process_paraphraser, i, responses_splits[i], args)
            for i in range(num_processes)
        ]
        temp_files = [future.result() for future in features]

    # all_file_path = [
    #     f"/media/5/yx/distill_watermark/data/paraphrase/pegasus/temp_{id}.json"
    #     for id in range(6)
    # ]
    # output_list = []
    # for file in all_file_path:
    #     with open(file, "r") as f:
    #         for line in f:
    #             output_list.append(json.loads(line))

    # 合并结果
    output_list = []
    for temp_file in temp_files:
        with open(temp_file, "r") as infile:
            for line in infile:
                json_obj = json.loads(line)
                output_list.append(json_obj)
        os.remove(temp_file)

    with open(output_file, "w") as f:
        json.dump(output_list, f, indent=4)

    print(f"\nGeneration completed. Results saved to: {output_file}")
