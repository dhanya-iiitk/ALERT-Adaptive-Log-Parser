---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:570
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 183.62.156.108:22 open through proxy socks.cse.cuhk.edu.hk:5070
    SOCKS5
  sentences:
  - 'Received disconnect from 183.62.140.253: 11: Bye Bye [preauth]'
  - 'i2.itc.cn:80 error : Could not connect through proxy proxy.cse.cuhk.edu.hk:5070
    - Proxy server cannot establish a connection with the target, status code 503'
  - 'Received disconnect from 187.141.143.180: 11: Bye Bye [preauth]'
- source_sentence: Accepted password for fztu from 119.137.62.142 port 49116 ssh2
  sentences:
  - 'pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser=
    rhost=103.99.0.122'
  - PAM 1 more authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=185.190.58.151
  - Failed none for invalid user admin from 5.188.10.180 port 52631 ssh2
- source_sentence: PAM service(sshd) ignoring max retries; 5 > 3
  sentences:
  - 'Received disconnect from 103.207.39.16: 11: Closed due to user request. [preauth]'
  - 'client.dropbox.com:443 error : Could not connect to proxy proxy.cse.cuhk.edu.hk:5070
    - Could not resolve proxy.cse.cuhk.edu.hk error 11001'
  - 'input_userauth_request: invalid user ubuntu [preauth]'
- source_sentence: 'config.pinyin.sogou.com:80 error : Could not connect through proxy
    proxy.cse.cuhk.edu.hk:5070 - Proxy closed the connection unexpectedly.'
  sentences:
  - 'qd-update.qq.com:8080 error : Could not connect through proxy proxy.cse.cuhk.edu.hk:5070
    - Proxy server cannot establish a connection with the target, status code 403'
  - 10.251.214.32:50010 Served block blk_-6520030462660619051 to /10.251.215.70
  - 'BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.122.65:50010 is added
    to blk_5095897972833086609 size 67108864'
- source_sentence: 'config.pinyin.sogou.com:80 error : Could not connect through proxy
    proxy.cse.cuhk.edu.hk:5070 - Proxy closed the connection unexpectedly.'
  sentences:
  - Connection closed by 194.190.163.22 [preauth]
  - 'bolt.dropbox.com:443 error : Could not connect to proxy proxy.cse.cuhk.edu.hk:5070
    - connection attempt failed with error 10061'
  - 'BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.7.230:50010 is added
    to blk_6238543795209416553 size 67108864'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'config.pinyin.sogou.com:80 error : Could not connect through proxy proxy.cse.cuhk.edu.hk:5070 - Proxy closed the connection unexpectedly.',
    'bolt.dropbox.com:443 error : Could not connect to proxy proxy.cse.cuhk.edu.hk:5070 - connection attempt failed with error 10061',
    'Connection closed by 194.190.163.22 [preauth]',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.1545, 0.2542],
#         [0.1545, 1.0000, 0.0174],
#         [0.2542, 0.0174, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 570 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 570 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                                                         |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | float                                                         |
  | details | <ul><li>min: 13 tokens</li><li>mean: 36.79 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 36.65 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                          | sentence_1                                                                                                                     | label            |
  |:--------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>BLOCK* ask 10.250.14.38:50010 to replicate blk_-7571492020523929240 to datanode(s) 10.251.122.38:50010</code> | <code>Received block blk_-5421365489363678155 of size 67108864 from /10.251.122.38</code>                                      | <code>0.0</code> |
  | <code>BLOCK* NameSystem.delete: blk_5018797996887008111 is added to invalidSet of 10.251.89.155:50010</code>        | <code>10.251.214.32:50010 Served block blk_-6520030462660619051 to /10.251.215.70</code>                                       | <code>0.0</code> |
  | <code>pam_unix(sshd:auth): check pass; user unknown</code>                                                          | <code>pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=183.62.140.253  user=root</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 1.3 minutes

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 5.4.1
- Transformers: 5.8.0
- PyTorch: 2.9.1+cu128
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->