<div align="center">
<h1>Fast Best-of-N Decoding via Speculative Rejection</h1>

**fast inference-time alignment**
</div>
</div>
<div align="center">
<b><a href="https://github.com/preminstrel">Hanshi Sun</a></b><sup>1*</sup>,
<b>Momin Haider</b><sup>2*</sup>,
<b><a href="https://rqzhangberkeley.github.io/">Ruiqi Zhang</a></b><sup>3*</sup>,
<b>Huitao Yang</b><sup>5</sup>,
<b><a href="https://ece.princeton.edu/people/jiahao-qiu">Jiahao Qiu</a></b><sup>4</sup>,
<br>
<b><a href="https://mingyin0312.github.io/">Ming Yin</a></b><sup>4</sup>,
<b><a href="https://mwang.princeton.edu/">Mengdi Wang</a></b><sup>4</sup>,
<b><a href="https://people.eecs.berkeley.edu/~bartlett/">Peter Bartlett</a></b><sup>3</sup>,
<b><a href="https://azanette.com/">Andrea Zanette</a></b><sup>1*</sup>
</div>
<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>University of Virginia
<sup>3</sup>UC Berkeley<br>
<sup>4</sup>Princeton University
<sup>5</sup>Fudan University<br>
<small>(* for core authors)</small>
</div><br>
<div align="center">
[<a href="XXXX">Paper</a>] | [<a href="https://Zanette-Labs.github.io/SpeculativeRejection">Blog</a>]
</div>
<br>

<div align="center">
<img src="static/images/spr.png" align="top"/>
</div>

## Environment Set Up
```bash
# create env
conda create -n SpecRej python=3.10 -y
conda activate SpecRej

# install packages
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

## Efficiency Evaluation
First, we need to run the Best-of-N baselines and Speculative Rejection. The following commands are examples of running the Best-of-120, Best-of-960, and Speculative Rejection (`alpha=0.5`) on the `Meta-Llama-3-8B` and `ArmoRM-Llama3-8B-v0.1`. For larger N (e.g., Best-of-3840), we can adjust the seed and merge the results from multiple runs using 8 H100 GPUs using `postprocess/concat_json.py`.
```bash
# Best-of-120
accelerate launch --num_processes 1 --num_machines 1 --gpu_ids 1 --machine_rank 0 --mixed_precision no --dynamo_backend no \
main.py --output_folder ./archive/Bo120_Meta-Llama-3-8B_ArmoRM-Llama3-8B-v0.1_0 \
--llm_name Meta-Llama-3-8B --reward_model_name ArmoRM-Llama3-8B-v0.1 \
--max_tokens 8000 --batch_size 120 --seed 0 

# ... (Best-of-240, Best-of-480)

# Best-of-960
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --gpu_ids 0,1,2,3,4,5,6,7 --machine_rank 0 --mixed_precision no \
--dynamo_backend no main.py --output_folder ./archive/Bo960_Meta-Llama-3-8B_ArmoRM-Llama3-8B-v0.1_0 \
--llm_name Meta-Llama-3-8B --reward_model_name ArmoRM-Llama3-8B-v0.1 \
--max_tokens 8000 --batch_size 120 --seed 0 

# Speculative Rejection (alpha=0.5)
accelerate launch --num_processes 1 --num_machines 1 --gpu_ids 0 --machine_rank 0 --mixed_precision no --dynamo_backend no \
main.py --output_folder ./archive/SpR_alpha_0.5_Meta-Llama-3-8B_ArmoRM-Llama3-8B-v0.1_0 \
--llm_name Meta-Llama-3-8B --reward_model_name ArmoRM-Llama3-8B-v0.1 \
--max_tokens 8000 --seed 0 \
--speculative_rejection --alpha 0.5
```

After gathering the results under `archive` folder, we can evaluate the efficiency of the Best-of-N baselines and Speculative Rejection using the following command.
```bash
# make sure the args correct in the script first
python postprocess/plot_compare.py
```


## Win-rate Evaluation

When we get the all the outputs from the Best-of-N baselines and Speculative Rejection, we can evaluate the win-rate using `alpaca_eval`.

First, we need to gather the best utterances from the outputs of the Best-of-N baselines and Speculative Rejection and merge the outputs for win-rate evaluation.

```bash
# gather best answers
python postprocess/gather_best_ans.py

# merge json files for win-rate evaluation
python postprocess/merge_json.py
```

Then, we can evaluate the win-rate using the following command.

```bash
export OPENAI_API_KEY=YOUR_API_KEY

alpaca_eval make_leaderboard --leaderboard_path leader_board.csv  --all_model_outputs win_rate/Meta-Llama-3-8B_ArmoRM-Llama3-8B-v0.1_compare.json   --reference_outputs win_rate/Meta-Llama-3-8B_ArmoRM-Llama3-8B-v0.1_ref.json --output_path leader_board --fn_metric 'get_length_controlled_winrate' --sort_by 'length_controlled_winrate'  --is_overwrite_leaderboard
```

## Citation
If you find Speculative Rejection useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{sun2024fast,
  title={Fast Best-of-N Decoding via Speculative Rejection},
  author={Sun, Hanshi and Haider, Momin and Zhang, Ruiqi and Yang, Huitao and Qiu, Jiahao and Yin, Ming and Wang, Mengdi and Bartlett, Peter and Zanette, Andrea},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2024}
}
```
