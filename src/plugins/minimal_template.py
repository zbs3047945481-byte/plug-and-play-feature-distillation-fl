from src.plugins.base import BaseClientPlugin, BaseServerPlugin


class MinimalClientPlugin(BaseClientPlugin):
    """Minimal integration template for external FL projects."""

    def __init__(self, options, model, gpu):
        self.options = options
        self.model = model
        self.gpu = gpu

    def on_round_start(self, learning_rate, server_payload):
        pass

    def train_batch(self, X, y):
        raise NotImplementedError

    def build_upload_payload(self):
        return None


class MinimalServerPlugin(BaseServerPlugin):
    """Minimal integration template for external FL projects."""

    def __init__(self, options, gpu):
        self.options = options
        self.gpu = gpu

    def build_broadcast_payload(self):
        return None

    def aggregate_client_payloads(self, local_model_paras_set):
        return None
