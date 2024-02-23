# Borrow idea from VoxPoser https://github.com/huangwl18/VoxPoser/blob/main/src/LMP.py
import os
import openai
from openai import RateLimitError, APIConnectionError
import json
import time
from time import sleep
from .cache import MyCache

RETRY_LIMIT = 0

MODEL_LIMIT = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]

with open("my_apikey", "r") as file:
    apikey = file.read().strip()

class MyGPT:
    def __init__(self, config_path="LLM/config/roboexp.json", cache_filename="roboexp"):
        # Read the GPT config from the config file
        with open(config_path, "r") as f:
            self.config = json.load(f)
        # Initialize the cache
        self._cache = MyCache(cache_filename)
        # Read the relevant parameters
        self.prompts = self._init_messages(self.config["messages"])

    def _init_messages(self, message_configs):
        # Initialize the messages
        messages = []
        for message in message_configs:
            content_path = message.pop("content_path")
            with open(content_path, "r") as f:
                message["content"] = f.read().strip()
            messages.append(message)
        return messages

    def _cached_api_call(
        self, query, model="gpt-3.5-turbo-1106", temperature=0, max_tokens=512, stop=None
    ):
        # For us, we don't need to further process the self.prompts
        # We only need to process the query
        message = {}
        message["role"] = "user"
        message[
            "content"
        ] = f"{self.config['query_prefix']}{query}{self.config['query_suffix']}"
        # Prpeare the parameters for the API call
        kwargs = {}
        kwargs["model"] = model
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
        if stop is not None:
            kwargs["stop"] = stop
        kwargs["messages"] = self.prompts + [message]
        # Check if the kwargs has been in the cache
        if kwargs in self._cache:
            response = self._cache[kwargs]["choices"][0]["message"]["content"]
            print(f"Using Cache: {response}")
        else:
            client = openai.OpenAI(api_key=apikey)
            result = client.chat.completions.create(**kwargs)
            result = result.model_dump()
            # # A dummy test
            # result = {
            #     "choices": [
            #         {
            #             "finish_reason": "stop",
            #             "index": 0,
            #             "message": {
            #                 "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
            #                 "role": "assistant",
            #             },
            #         }
            #     ],
            #     "created": 1677664795,
            #     "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
            #     "model": "gpt-3.5-turbo-1106",
            #     "object": "chat.completion",
            #     "usage": {
            #         "completion_tokens": 17,
            #         "prompt_tokens": 57,
            #         "total_tokens": 74,
            #     },
            # }
            self._cache[kwargs] = result
            response = result["choices"][0]["message"]["content"]
            print(f"Using API: {response}")
        return response

    def __call__(self, query):
        # Check if the model is in my list
        assert self.config["model"] in MODEL_LIMIT
        start_time = time.time()
        current_retry = 0
        while True:
            try:
                response = self._cached_api_call(
                    query,
                    self.config["model"],
                    self.config["temperature"],
                    self.config["max_tokens"],
                    self.config["stop"],
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                current_retry += 1
                if current_retry > RETRY_LIMIT:
                    break
                else:
                    print(f"OpenAI API got err {e}")
                    print("Retrying after 3s.")
                    sleep(3)
        print(f"*** OpenAI API call took {time.time() - start_time:.2f}s ***")
        return response
