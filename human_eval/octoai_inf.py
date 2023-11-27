import requests
import os
import json
import concurrent.futures

import time

from lm_eval.base import BaseLM
import openai
REPEAT_REQUEST_TO_OCTOAI_SEREVER = 10
B_INST, E_INST = "[INST]", "[/INST]"

model_urls = {
    "codellama-34b-instruct-int4":	"https://text.customer-endpoints.nimbus.octoml.ai",
    "llama-2-70b-chat-int4":	"https://text.customer-endpoints.nimbus.octoml.ai",

    "codellama-13b-instruct-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",
    "codellama-34b-instruct-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",
    "codellama-7b-instruct-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",

    "llama-2-70b-chat-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",
    "llama-2-13b-chat-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",
    
    "mistral-7b-instruct-fp16":	"https://text.customer-endpoints.nimbus.octoml.ai",

    
    
}
model_urls_prod = {
    "codellama-34b-instruct-int4":	"https://text.octoai.run/v1",
    "llama-2-70b-chat-int4":	"https://text.octoai.run/v1",

    "codellama-13b-instruct-fp16":	"https://text.octoai.run/v1",
    "codellama-34b-instruct-fp16":	"https://text.octoai.run/v1",
    "codellama-7b-instruct-fp16":	"https://text.octoai.run/v1",

    "llama-2-70b-chat-fp16":	"https://text.octoai.run/v1",
    "llama-2-13b-chat-fp16":	"https://text.octoai.run/v1",
    
    "mistral-7b-instruct-fp16":	"https://text.octoai.run/v1",
}

class OctoAIEndpointLM(BaseLM):
  def __init__(
      self,
      model_name="llama-2-70b-chat",
      batch_size=1,
      max_batch_size=None,
      device=None,
      use_prod=None):
    """
    :param model_name: str
        Model name from the list of models supported by OctoAI
    """
    super().__init__()

    self.time_meas = True

    self.model_name = model_name
    self._batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self._device=device
    self.use_prod = use_prod

    self.init_remote()

  def init_remote(self):
    # TODO(vvchernov): possibly not-safe approach need to get key each time
    # Get the API key from the environment variables
    if self.use_prod:
      api_key=os.environ["OCTOAI_API_KEY_PROD"]
    else:
      api_key=os.environ["OCTOAI_API_KEY"]

    if api_key is None:
      raise ValueError("API_KEY not found in the .env file")

    if self.use_prod:
       self.url = model_urls_prod[self.model_name]
       openai.api_base = self.url 
    else:
      self.url = model_urls[self.model_name]
      openai.api_base = self.url + "/v1"

    openai.api_key = api_key
    self.headers = {
      "accept": "text/event-stream",
      "authorization": f"Bearer {api_key}",
      "content-type": "application/json",
    }
    self.data_template = {
        "model": self.model_name,
        "messages": [
            {
                "role": "assistant",
                # "content":"generate only python function and remove comments of this function, and remove your explanations and mark all the code with [PYTHON] at the beginning of the code and [/PYTHON] at the end of the code for: \n"
                "content": "generate only python code and remove comments for:"
            }
        ],
        "stream": False,
        "max_tokens": 256
    }

  @property
  def eot_token_id(self):
    raise NotImplementedError("No idea about anthropic tokenization.")

  @property
  def max_length(self):
    return 2048

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def device(self):
    return self._device

  def tok_encode(self, string: str):
      return string
      #raise NotImplementedError("No idea about anthropic tokenization.")

  def tok_decode(self, tokens):
      return tokens
      #raise NotImplementedError("No idea about anthropic tokenization.")

  def _loglikelihood_tokens(self, requests, disable_tqdm=False):
    raise NotImplementedError("No support for logits.")

  def greedy_until(self, requests):
        if not requests:
            return []

        results = []
        if self.time_meas:
            start_timer = time.time()
        if self.batch_size > 1:
            def _batcher(in_requests):
                for i in range(0, len(in_requests), self.batch_size):
                    yield in_requests[i:i + self.batch_size]

            for request_batch in _batcher(requests):
                try:
                    self._model_generate_parallel(request_batch, results)
                except ConnectionError as e:
                    print(f"ConnectionError: {e}. Skipping this batch and continuing...")

        else:
            for request in requests:
                # inp = request[0]
                # request_args = request[1]
                # until = request_args["until"]
                try:
                    self._model_generate(request, results)
                except ConnectionError as e:
                    print(f"ConnectionError: {e}. Skipping this request and continuing...")

        if self.time_meas:
            stop_timer = time.time()
            secs = stop_timer - start_timer
            print(
                "Full time of predictions measurement: {:.2f} sec, {:.2f} min, {:.2f} hour(s)".format(
                    secs, secs / 60, secs / 3600))

        return results

  def call_octoai_inference(self, user_input: str):
    import copy
    data = copy.deepcopy(self.data_template)
    data["messages"][0]["content"] += user_input

    completion = openai.ChatCompletion.create(model=self.model_name, messages=data["messages"], max_tokens=300)
    # response = requests.post(self.url + "/v1/chat/completions", headers=self.headers, json=data)
    # if response.status_code != 200:
    #   print(f"Error: {response.status_code} - {response.text}")

    return completion

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")
  
  def call_octoai_reset(self):
    try:
      resp = requests.post(self.url + "/chat/reset", headers = self.headers)
      return resp.json()
    except Exception as e:
      print(f"Error resetting chat for endpoint {self.url}")
      print(e)
      return
    
  # TODO(vvchernov): do we need additional args? max_tokens, temperature..
  def _model_generate(self, inps, results, stop=[]):
    success = False
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SEREVER):
      response = self.call_octoai_inference(inps)
      response = json.loads(response.text)
      if 'choices' in response.keys():
        success = True
        break
    if success:
      #print(response['choices'][0]['message']['content'])
      results.append(response['choices'][0]['message']['content'])
      # print(response['choices'][0]['message']['content'])
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append("Dummy response")

  def _model_generate_parallel(self, request_batch, results=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results={}
      for task_id in request_batch:
        parallel_results[task_id]=[]
        inp = request_batch[task_id]["prompt"]
        # until = request_args["until"]
        futures.append(executor.submit(self._model_generate, inp, parallel_results[task_id]))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")
          #raise RuntimeError(f"Error parallel generating predictions: {exc}")

      return parallel_results
      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])