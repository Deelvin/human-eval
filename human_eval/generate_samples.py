from human_eval.data import write_jsonl, read_problems
from octoai_human_eval import OctoAIEndpointLM
import json
import itertools
import re

MODEL_NAMES = [ "codellama-7b-instruct-mlc-q0f16",
                "codellama-7b-instruct-mlc-q4f16_1",
                "codellama-7b-instruct-mlc-q8f16_1",
                "codellama-13b-instruct-mlc-q0f16",
                "codellama-13b-instruct-mlc-q4f16_1",
                "codellama-13b-instruct-mlc-q8f16_1",
                "codellama-34b-instruct-mlc-q0f16",
                "codellama-34b-instruct-mlc-q4f16_1",
                "codellama-34b-instruct-mlc-q8f16_1",
                "llama2-7b-chat-mlc-q0f16",
                "llama2-7b-chat-mlc-q4f16_1",
                "llama2-7b-chat-mlc-q8f16_1",
                "llama2-13b-chat-mlc-q0f16",
                "llama2-13b-chat-mlc-q4f16_1",
                "llama2-13b-chat-mlc-q8f16_1",
                "llama2-70b-chat-mlc-q0f16",
                "llama2-70b-chat-mlc-q4f16_1",
                "llama2-70b-chat-mlc-q8f16_1"
               ]



def extract_output(output):
    out = output.split("[/INST]")[-1].split("</s>")[0].strip()
    out = out.replace('```', '')
    return out


# works well for codellama
def extract_code(string):
    out = string.replace('```', '')
    try:
        code_idx_start = out.index('[PYTHON]')
        out = "\n" + out[code_idx_start + len('[PYTHON]'):]
        code_idx_end = out.index('[/PYTHON]')
        out = out[:code_idx_end]

    except ValueError:
        return out
    except IndexError:
        return out


    return extract_output(out)

# for some models, especially for Lamma 13-b, this post-processing works better
def extract_code_llama(string):
    out = string.replace('```', '')
    code_idx_start = out.index('\n')
    out = out[code_idx_start:]
    try:
        code_idx_end = [i for i in range(len(out)) if out.startswith('    return', i)][-1]
        while (out[code_idx_end] != '\n') and (code_idx_end < len(out) - 1):
            code_idx_end += 1
        return out[:code_idx_end + 1]        
    except IndexError:
        return out
    

def generate_one_completion(inp,):
    tmp = endpoint.call_octoai_reset()
    tmp = endpoint.call_octoai_inference(inp)
    tmp = json.loads(tmp.text)
    tmp = tmp['choices'][0]['message']['content'] 
    out = extract_code(tmp)
    return out

problems = read_problems()
import time
import os
import tqdm
for model_name in MODEL_NAMES:
    samples = []
    num_samples_per_task = 1
    endpoint = OctoAIEndpointLM(model_name)
    gen_start_time = time.time()
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in tqdm.tqdm(problems)
    ]
    gen_end_time = time.time()
    model_name += "_reset"
    os.mkdir(f"results/{model_name}")
    write_jsonl(f"results/{model_name}/samples.jsonl", samples)
    tmp = os.popen(f"evaluate_functional_correctness results/{model_name}/samples.jsonl").read()
    with open(f"results/{model_name}/time_stat.txt", 'w') as f:
        f.write(str((gen_end_time - gen_start_time) / 60))
        f.write(tmp.split('\n')[-2])