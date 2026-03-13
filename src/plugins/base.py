from abc import ABC, abstractmethod


class BaseClientPlugin(ABC):
    @abstractmethod
    def on_round_start(self, learning_rate, server_payload):
        pass

    @abstractmethod
    def train_batch(self, X, y):
        pass

    @abstractmethod
    def build_upload_payload(self):
        pass


class BaseServerPlugin(ABC):
    @abstractmethod
    def build_broadcast_payload(self):
        pass

    @abstractmethod
    def aggregate_client_payloads(self, local_model_paras_set):
        pass
