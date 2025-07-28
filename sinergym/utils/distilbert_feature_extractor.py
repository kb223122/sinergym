import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from transformers import DistilBertModel, DistilBertTokenizerFast

class DistilBertFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses DistilBERT to process observations.
    This is a demonstration. It assumes observations are 1D float arrays that can be mapped to token IDs.
    In real use, you may need to preprocess observations to text or categorical tokens.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # For demonstration, we convert float obs to string tokens and use tokenizer
        # In practice, you should design a better mapping for your use case
        batch = []
        for obs in observations:
            # Convert each float to a string token (e.g., '23.5')
            tokens = [str(float(x.item())) for x in obs]
            # Join as a space-separated string
            text = ' '.join(tokens)
            batch.append(text)
        # Tokenize
        encoding = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=32)
        encoding = {k: v.to(observations.device) for k, v in encoding.items()}
        # Pass through BERT
        outputs = self.bert(**encoding)
        # Use the [CLS] token representation (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        # Project to features_dim
        features = self.fc(cls_embeddings)
        return features