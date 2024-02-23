import os
import json
import hashlib
from roboexp.utils import get_current_time


class MyCache:
    def __init__(self, base_path, filename="roboexp"):
        # Use json to cache all results from the conversation
        # Create the file if not existed, if existed, create the backup to store the results before modifying
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.cache_filepath = f"{base_path}/{filename}.json"
        self.temp_cache_filepath = f"{base_path}/{filename}_temp.json"
        self.cache = {}
        if os.path.exists(self.cache_filepath):
            # Backup the cache before the experiments
            os.system(
                f"cp {self.cache_filepath} {base_path}/{filename}_backup_{get_current_time()}.json"
            )
            with open(self.cache_filepath, "r") as f:
                self.cache = json.load(f)

    def save(self):
        # Back up the cache during the experiments, need to save the current history into a temp first, in case there are some errors
        if os.path.exists(self.cache_filepath):
            os.system(f"cp {self.cache_filepath} {self.temp_cache_filepath}")
        with open(self.cache_filepath, "w") as f:
            json.dump(self.cache, f)

    def __setitem__(self, key, value):
        # Add the key-value pair into the cache
        # The key is the hash of the key
        # Save the stuff everytime the cache is updated
        key_hash = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()
        self.cache[key_hash] = (key, value)
        self.save()

    def __getitem__(self, key):
        # Return the value
        key_hash = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()
        if key_hash in self.cache:
            return self.cache[key_hash][1]
        else:
            raise KeyError(f"Key {key} not in the cache")

    def __contains__(self, key):
        key_hash = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()
        return key_hash in self.cache

    def __delitem__(self, key):
        key_hash = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()
        if key_hash in self.cache:
            del self.cache[key_hash]
            self.save()
        else:
            raise KeyError(f"Key {key} not in the cache")

    def __len__(self):
        return len(self.cache)
