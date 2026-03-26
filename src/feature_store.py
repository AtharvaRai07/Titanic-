import redis
import json

class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def _entity_key(self, entity_id):
        # Normalize IDs so 332 and 332.0 map to the same Redis key.
        try:
            entity_id = int(float(entity_id))
        except (TypeError, ValueError):
            entity_id = str(entity_id)
        return f"features:{entity_id}:features"

    def store_feature(self, entity_id, features):
        key = self._entity_key(entity_id)
        self.client.set(key, json.dumps(features))

    def retrieve_feature(self, entity_id):
        key = self._entity_key(entity_id)
        features = self.client.get(key)
        if features:
            return json.loads(features)
        return None

    def store_batch_feature(self, batch_data):
        for entity_id, features in batch_data.items():
            self.store_feature(entity_id, features)

    def retrieve_batch_feature(self, entity_ids):
        batch_features = {}
        for entity_id in entity_ids:
            feature = self.retrieve_feature(entity_id)
            if feature:
                batch_features[entity_id] = feature
        return batch_features

    def get_all_entity_ids(self):
        keys = self.client.keys("features:*:features")
        entity_ids = [key.split(":")[1] for key in keys]
        return entity_ids

