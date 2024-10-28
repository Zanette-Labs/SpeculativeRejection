import torch


class KV_Cache:
    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 256,
        device: str = "cuda:0",
        dtype=torch.float16,
    ) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype,
        )
        self.batch_size = batch_size
        self.num_layers = config.num_hidden_layers
        self.kv_offset = torch.zeros(batch_size, dtype=torch.int32).to(self.device)
        self.head_dim = config.hidden_size // config.num_attention_heads

    def __str__(self):
        return f"[KV Cache] bsz-{self.batch_size} | layer-{self.num_layers} | max_length-{self.max_length} |head_dim-{self.head_dim} | {self.device} {self.dtype}"

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        storage_ids: torch.LongTensor,
    ):

        indices_expanded = (
            storage_ids.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.config.num_key_value_heads, self.head_dim)
        )
        self.k_cache[layer_idx].scatter_(1, indices_expanded, new_k_cache)
        self.v_cache[layer_idx].scatter_(1, indices_expanded, new_v_cache)

        if layer_idx == 0:
            self.kv_offset = self.kv_offset + new_k_cache.shape[-3]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset.zero_()

    def get_kv_len(self):
        return self.kv_offset
