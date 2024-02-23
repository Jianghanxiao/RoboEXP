# Borrow idea from VoxPoser https://github.com/huangwl18/VoxPoser/blob/main/src/LMP.py
import openai
from openai import RateLimitError, APIConnectionError
import json
import time
from time import sleep
from .cache import MyCache
import base64
import re

RETRY_LIMIT = 3

MODEL_LIMIT = ["gpt-4-vision-preview"]


class MyGPTV:
    def __init__(
        self, config_path, base_path, cache_filename="roboexp", REPLAY_FLAG=False
    ):
        if not REPLAY_FLAG:
            with open("my_apikey", "r") as file:
                self.apikey = file.read().strip()
        # Read the GPT config from the config file
        with open(config_path, "r") as f:
            self.config = json.load(f)
        # Initialize the cache
        self._cache = MyCache(base_path=base_path, filename=cache_filename)
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
        self,
        query,
        query_image_paths,
        model="gpt-4-vision-preview",
        temperature=0,
        max_tokens=300,
        stop=None,
    ):
        # Process the query images
        query_images = [
            self._encode_image(image_path) for image_path in query_image_paths
        ]
        # We only need to process the query
        message = {}
        message["role"] = "user"
        message["content"] = [
            {
                "type": "text",
                "text": f"{self.config['query_prefix']}{query}{self.config['query_suffix']}",
            },
        ]
        for query_image in query_images:
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{query_image}",
                        "detail": "auto",
                    },
                }
            )
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
            print("Using Cache")
            response = self._cache[kwargs]["choices"][0]["message"]["content"]
            print(f"Using Cache: {response}")
        else:
            try_count = 0
            while True:
                print("Using API")
                client = openai.OpenAI(api_key=self.apikey)
                result = client.chat.completions.create(**kwargs)
                result = result.model_dump()
                response = result["choices"][0]["message"]["content"]
                print(f"Using API: {response}")
                if re.search(r"\[Final Answer\]:\s*(.*)", response) is not None:
                    self._cache[kwargs] = result
                    break
                else:
                    try_count += 1
                    if try_count > RETRY_LIMIT:
                        raise ValueError(f"Over Limit: Unknown response: {response}")
                    else:
                        print("Retrying after 1s.")
                        sleep(1)
        return response

    def __call__(self, query, query_image_paths=[]):
        # Check if the model is in my list
        assert self.config["model"] in MODEL_LIMIT
        start_time = time.time()
        current_retry = 0
        while True:
            try:
                response = self._cached_api_call(
                    query,
                    query_image_paths,
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

    # Function to encode the image
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
