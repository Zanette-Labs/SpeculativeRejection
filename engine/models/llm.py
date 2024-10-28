import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import torch.nn.functional as F
import gc
import time
import pandas as pd
import numpy as np
import math
import vllm

from flash_attn import flash_attn_with_kvcache

from .kv_cache import KV_Cache
from engine.utils.sampling import norm_logits, sample
from engine.utils.info import gpu_memory, gpu_free_memory

from flashinfer.norm import rmsnorm
def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    return rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(
        hidden_states
    )


class LlamaLayer:
    def __init__(self, layer_idx) -> None:

        self.wqkv: torch.Tensor = None
        self.wo: torch.Tensor = None

        self.gate_up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wqkv: torch.Tensor = torch.cat(
            (
                hf_layer.self_attn.q_proj.weight.detach(),
                hf_layer.self_attn.k_proj.weight.detach(),
                hf_layer.self_attn.v_proj.weight.detach(),
            ),
            dim=0,
        )
        self.wo: torch.Tensor = hf_layer.self_attn.o_proj.weight.detach()
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        self.gate_up_proj = torch.cat(
            (
                hf_layer.mlp.gate_proj.weight.detach(),
                hf_layer.mlp.up_proj.weight.detach(),
            ),
            dim=0,
        )
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = (
            hf_layer.input_layernorm.variance_epsilon
        )

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = (
            hf_layer.post_attention_layernorm.variance_epsilon
        )

    def init_gpu(self, device: str = "cuda:0"):

        self.input_layernorm_weight = self.input_layernorm_weight.to(
            device, non_blocking=True
        )
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(
            device, non_blocking=True
        )
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj = self.down_proj.to(device, non_blocking=True)


class LLM:
    def __init__(
        self,
        model_name: str = "hmomin/sft10k",
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    ) -> None:

        self.local_files_only = local_files_only
        print(f"Initializing LLM with {model_name} on {device} with dtype: {dtype}")
        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, legacy=False, local_files_only=local_files_only
        )
        self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.softmax_scale = 1 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=self.dtype, device=self.device)
        )

        torch.cuda.synchronize()
        model_hbm = gpu_memory(self.device)
        print(f"[model w/o cache init on {device}]:  {model_hbm}")

        self.prefill_phase = False

    def __str__(self) -> str:
        return f"LLM: {self.model_name}, device: {self.device}, dtype: {self.dtype}"

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(
            self.config.max_position_embeddings,
            device=self.device,
            dtype=inv_freq.dtype,
        )
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        )
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        try:
            self.cos_cache = (
                hf_model.model.layers[0]
                .self_attn.rotary_emb.cos_cached.to(self.device)
                .to(self.dtype)
            )
            self.sin_cache = (
                hf_model.model.layers[0]
                .self_attn.rotary_emb.sin_cached.to(self.device)
                .to(self.dtype)
            )
        except:
            print("RoPE cache not found, initializing RoPE cache")
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache(
                hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.to(self.device)
            )

            # for vllm
            self.cos_sin_cache = torch.cat(
                (self.cos_cache[:, :64], self.sin_cache[:, :64]), dim=-1
            )

        self.layers: list[LlamaLayer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: LlamaLayer,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        bsz = hidden_states.shape[0]
        hidden_states = layer_norm(
            hidden_states,
            buffer.input_layernorm_variance_epsilon,
            buffer.input_layernorm_weight,
        )
        qkv = F.linear(hidden_states, buffer.wqkv)
        query_states, key_states, value_states = qkv.split(
            [buffer.q_size, buffer.kv_size, buffer.kv_size], dim=-1
        )
        return (
            query_states,
            key_states,
            value_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2),
        )

    def post_attention_compute(
        self, attn_output: torch.Tensor, residual: torch.Tensor, buffer: LlamaLayer
    ):
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(
            hidden_states,
            buffer.post_attention_layernorm_variance_epsilon,
            buffer.post_attention_layernorm_weight,
        )

        hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
        d = hidden_states.shape[-1] // 2
        output_shape = hidden_states.shape[:-1] + (d,)
        out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        vllm._custom_ops.silu_and_mul(out, hidden_states)

        hidden_states = F.linear(out, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def layer_compute_wo_cache(
        self,
        buffer: LlamaLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
    ):
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )

        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states.transpose(1, 2),
            v_cache=value_states.transpose(1, 2),
            causal=True,
        )

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

        hidden_states = self.post_attention_compute(
            hidden_states,
            residual,
            buffer,
        )

        return hidden_states

    @torch.inference_mode()
    def layer_compute(
        self,
        buffer: LlamaLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        storage_ids: torch.LongTensor,
    ):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, position_ids
        )
        key_states, value_states = self.kv_cache.update_kv_cache(
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            layer_idx,
            storage_ids,
        )

        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states,
            v_cache=value_states,
            cache_seqlens=self.kv_cache.kv_offset,
            causal=True,
        )

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

        hidden_states = self.post_attention_compute(
            hidden_states,
            residual,
            buffer,
        )

        return hidden_states

    @torch.inference_mode()
    def inference(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        storage_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        return_logits=True,
    ):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(
                self.layers[idx],
                idx,
                hidden_states,
                position_ids,
                attention_mask,
                storage_ids,
            )

        hidden_states = layer_norm(
            hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon
        )

        if self.prefill_phase:  # prefill
            hidden_states = hidden_states[:, -1:, :]

        if return_logits:
            logits = F.linear(hidden_states, self.lm_head).float()
            return logits

    @torch.inference_mode()
    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        vllm._custom_ops.rotary_embedding(
            position_ids, q, k, 128, self.cos_sin_cache, True
        )
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(
            input_len, dtype=torch.long, device=self.device
        ) + past_len.unsqueeze(1)
        storage_ids = position_ids.clone()
        return position_ids, storage_ids

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()

        iter_prefill = max((input_ids.shape[0] * input_ids.shape[1]) // (5000), 1)
        T = (input_ids.shape[1] + iter_prefill - 1) // iter_prefill
        iter_prefill = (input_ids.shape[1] + T - 1) // T
        for i in range(iter_prefill):
            if input_ids[:, i * T : (i + 1) * T].shape[-1] < 1:
                print("break")
                break
            try:
                position_ids, storage_ids = self.get_ctx(
                    input_ids[:, i * T : (i + 1) * T]
                )
                if i == iter_prefill - 1:
                    logits = self.inference(
                        input_ids=input_ids[:, i * T : (i + 1) * T],
                        position_ids=position_ids,
                        attention_mask=None,
                        storage_ids=storage_ids,
                    )
                else:
                    self.inference(
                        input_ids=input_ids[:, i * T : (i + 1) * T],
                        position_ids=position_ids,
                        attention_mask=None,
                        storage_ids=storage_ids,
                        return_logits=False,
                    )
            except Exception as e:
                print(
                    position_ids,
                    storage_ids,
                    input_ids.shape,
                    i,
                    T,
                    input_ids[:, i * T : (i + 1) * T].shape,
                    self.kv_cache.kv_offset,
                )
                raise e

        return logits

    def encode(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return input_ids

    def batch_encode(self, text_list):
        input_ids = self.tokenizer(
            text_list, return_tensors="pt", padding=True, add_special_tokens=False
        ).input_ids.to(self.device)

        min_length = min(
            input_ids.shape[1],
            torch.min(
                torch.sum(input_ids != self.tokenizer.pad_token_id, dim=1)
            ).item(),
        )
        input_ids = input_ids[:, :min_length]
        return input_ids

    def decode(self, input_ids: torch.LongTensor):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def init_kv_cache(self, max_length: int = 256):
        if (
            "TinyLlama" in self.model_name
            or "JackFram" in self.model_name
            or "68" in self.model_name
        ):
            if not hasattr(self, "gamma"):
                self.gamma = 4
            self.kv_cache = SinkCache(
                self.config,
                max_length=max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                gamma=self.gamma,
            )
        else:
            self.kv_cache = KV_Cache(
                self.config,
                max_length=max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
            )
        torch.cuda.synchronize()
        model_kv_cache_hbm = gpu_memory(self.device)
        # print(self.kv_cache)
        print(
            f"[model ({self.model_name}) w/ cache init on {self.device}]:  {model_kv_cache_hbm}"
        )

    def get_gen_len(self, batch_size: int, cur_len: int):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem_GB = gpu_free_memory(self.device) - 15  # 4GB buffer
        # for example, (1, 32K) seq len = 16GB for sft10k
        max_seq = int(free_mem_GB * 1024 * 2 / batch_size * (self.num_key_value_groups))
        max_seq = min(max_seq, self.max_tokens)
        # assert max_seq > cur_len, f"Max sequence length {max_seq} is less than input length {cur_len}"
        return max_seq - cur_len

    def get_batch_size(self, max_seq: int):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem_GB = gpu_free_memory(self.device) - 4
        batch_size = int(free_mem_GB * 1024 * 2 / max_seq * (self.num_key_value_groups))
        return batch_size


    @torch.inference_mode()
    def inference_wo_cache(self, input_ids: torch.LongTensor):
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        input_len = input_ids.shape[-1]
        position_ids = torch.arange(
            input_len, dtype=torch.long, device=self.device
        )

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute_wo_cache(
                self.layers[idx],
                idx,
                hidden_states,
                position_ids,
            )

        hidden_states = layer_norm(
            hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon
        )

        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def self_evaluate(self, input_ids: torch.LongTensor):
        score = []
        
        T = 1
        num_iter = (input_ids.shape[0] + T - 1) // T

        for i in range(num_iter):
            input_ = input_ids[i * T : (i + 1) * T]
            logits = self.inference_wo_cache(input_)

            loss = None

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # [T, seq, vocab_size]
            shift_labels = input_ids[i * T : (i + 1) * T, 1:].contiguous() # [T, seq]
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Enable model parallelism
            loss = -torch.exp(loss_fct(shift_logits, shift_labels).view(T, -1).mean(dim=-1))
            score.append(loss.tolist())
        
        return score

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1024,
        gen_len: int = 256,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        benchmark: bool = False,
    ):

        self.batch_size = batch_size

        if type(input_ids) == str:
            input_ids = self.encode(input_ids)

        if input_ids.size(0) != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: {input_ids.size(0)} != {self.batch_size}"
            )

        if benchmark:
            torch.cuda.synchronize()
            start = time.time()

        # init kv cache
        max_length = input_ids.size(1) + gen_len
        self.init_kv_cache(max_length=max_length)

        # prefill
        self.prefill_phase = True
        logits = self.prefill(input_ids)
        self.prefill_phase = False

        next_token = sample(
            norm_logits(
                logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k
            )
        )

        if benchmark:
            torch.cuda.synchronize()
            prefill = time.time()

        n = 0

        generated_ids = [[] for _ in range(self.batch_size)]
        for i, token in enumerate(next_token.tolist()):
            generated_ids[i].extend(input_ids[i].tolist())
            generated_ids[i].extend(token)

        finished_generations = torch.zeros(batch_size).bool().to(self.device)
        while n < gen_len:
            position_ids, storage_ids = self.get_ctx(next_token)
            attention_mask = None

            logits = self.inference(
                input_ids=next_token,
                position_ids=position_ids,
                attention_mask=attention_mask,
                storage_ids=storage_ids,
            )
            next_token = sample(
                norm_logits(
                    logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k
                )
            )

            n += 1
            for i, token in enumerate(next_token.tolist()):
                generated_ids[i].extend(token)

            just_finished = (
                torch.LongTensor(np.array(generated_ids))[:, -1].to(self.device)
                == self.tokenizer.pad_token_id
            )
            finished_generations = finished_generations | just_finished
            if torch.all(finished_generations):
                break

        if benchmark:
            torch.cuda.synchronize()
            end = time.time()
            print(
                f"Time taken: {end-prefill} to generate {gen_len} tokens, Prefill time: {prefill-start} ({input_ids.shape})"
            )

        # free KV Cache
        del self.kv_cache
        self.kv_cache = None
        gc.collect()
        torch.cuda.empty_cache()

        return torch.LongTensor(np.array(generated_ids)).to(self.device)

