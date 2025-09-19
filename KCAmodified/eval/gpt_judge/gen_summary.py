"""Generate model answer for summary task."""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray
import datasets

from fastchat.model import get_conversation_template

#new modules
import evaluate
from openai import OpenAI, AsyncOpenAI


def compute_metrics(ans_jsons, no_sorry):
    #rouge = datasets.load_metric('rouge') #adjusting to updated modules
    rouge = evaluate.load("rouge")
    if no_sorry:
        print("Delete sorry answers...")
        old_len = len(ans_jsons)
        ans_jsons = [line for line in ans_jsons if "Sorry, I don't know the factual information required to answer this question." not in line["answer"]]
        print(f"Delete {old_len - len(ans_jsons)} examples.")
    predictions = [line["answer"] for line in ans_jsons]
    references = [line["reference_answers"][-1] for line in ans_jsons]
    rouge_results = rouge.compute(predictions=predictions, references=references)
    return {#"ROUGE-1": round(rouge_results["rouge1"].mid.fmeasure * 100, 2), #adjusted for compatibility with new modules
            "ROUGE-1": round(rouge_results["rouge1"] * 100, 2),
            #"ROUGE-2": round(rouge_results["rouge2"].mid.fmeasure * 100, 2),
            "ROUGE-2": round(rouge_results["rouge2"] * 100, 2),
            #"ROUGE-L": round(rouge_results["rougeL"].mid.fmeasure * 100, 2),
            "ROUGE-L": round(rouge_results["rougeL"] * 100, 2),
            #"ROUGE-Lsum": round(rouge_results["rougeLsum"].mid.fmeasure * 100, 2),
            "ROUGE-Lsum": round(rouge_results["rougeLsum"] * 100, 2),
            }


def run_eval(model_path, model_id, conv_temp, question_file, answer_file, metric_file, num_gpus, do_sample=False, no_sorry=False):
    if question_file.endswith(".jsonl"):
        ques_jsons = []
        with open(os.path.expanduser(question_file), "r") as ques_file:
            for line in ques_file:
                ques_jsons.append(json.loads(line))
    else:
        ques_jsons = json.loads(open(question_file, "r").read())
    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers.remote(
                model_path, model_id, conv_temp, ques_jsons[i: i + chunk_size], do_sample
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

    evaluation_results = compute_metrics(ans_jsons, no_sorry)
    print(json.dumps(evaluation_results, indent=2))

    if metric_file:
        #os.makedirs(metric_file.replace("metrics.json", ""), exist_ok=True) #commented out to avoid errors; simply moving the file proved easier

        with open(os.path.expanduser(metric_file), "w") as met_file:
            json.dump(evaluation_results, met_file, indent=2)


@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, conv_temp, question_jsons, do_sample=False):
    model_path = os.path.expanduser(model_path)
    #modifying for ollama alternative
    if "Tiny-LLM-llama" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False if "pythia" not in model_path else True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()
    
    
    #adding in ollama alternative
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = line
        idx = ques_json["idx"]
        qs = ques_json["Instruction"]
        conv = get_conversation_template(conv_temp)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #modifying for ollama alterantive
        if "Tiny-LLM-llama" in model_path:
            input_ids = tokenizer([prompt]).input_ids
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=do_sample,
                temperature=0.7,
                max_new_tokens=1024,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        else:
            client.additional_kwargs={"temperature":0.7, "max_new_tokens":1024}
            response = client.chat.completions.create(
                model="qwen2:1.5b",
                messages=[{"role": "user", "content": str(prompt)}],
                )
            outputs = response.choices[0].message.content
        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "id": idx,
                "class": "summary",
                "question": qs,
                "reference_answers": ques_json["Reference_Answers"],
                "answer": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {"do_sample": do_sample, "temperature": 0.7, "max_new_tokens": 1024},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen2:1.5b", required=False) #or: Tiny-LLM-llama
    parser.add_argument("--model-id", type=str, default="llama", required=False)
    parser.add_argument("--conv-temp", type=str, default="vicuna")
    parser.add_argument("--question-file", type=str, default="~/Downloads/Projekt/KCA/data/metric_based_evaluation/msmacro_testset.jsonl", required=False) #or: acibench
    parser.add_argument("--answer-file", type=str, default="~/Downloads/Projekt/KCA/evaluation_results/answer_greedy/data-Qwen2:1.5b_shot-5_msmacro_model-llama_greedy.jsonl")
    parser.add_argument("--metric-file", type=str, default="metrics.json")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-sorry", action="store_true")
    args = parser.parse_args()

    ray.init()
    run_eval(
        args.model_path,
        args.model_id,
        args.conv_temp,
        args.question_file,
        args.answer_file,
        args.metric_file,
        args.num_gpus,
        args.do_sample,
        args.no_sorry,
    )
