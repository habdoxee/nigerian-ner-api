import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunedEmbedding(nn.Module):
    """Small fp16 embedding table plus a map from original token ids to rows."""

    def __init__(self, old_vocab, new_vocab, hidden):
        super().__init__()
        self.padding_idx = 1
        self.register_buffer("id_map", torch.zeros(old_vocab, dtype=torch.long))
        self.weight = nn.Parameter(
            torch.zeros(new_vocab, hidden, dtype=torch.float16), requires_grad=False
        )

    def forward(self, input_ids):
        return F.embedding(self.id_map[input_ids], self.weight).float()


def load_pruned_model(path):
    from transformers import AutoConfig, AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(path)
    state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    key = next(k for k in state if k.endswith("word_embeddings.id_map"))
    old_vocab = state[key].shape[0]
    model = AutoModelForTokenClassification.from_config(config)
    model.base_model.embeddings.word_embeddings = PrunedEmbedding(
        old_vocab, config.vocab_size, config.hidden_size
    )
    model.load_state_dict(state)
    model.eval()
    return model
