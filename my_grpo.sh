#!/bin/bash

# =========================
# 配置：
# Search-R1 Conservative Stable Config (4x RTX 4090 24GB)
# 常用命令：
# 1. 训练：conda activate searchr1 && bash my_grpo.sh
# 2. 启动检索服务器：conda activate retriever && bash retrieval_launch.sh
# 3. 实时显卡监控（只看显存）：watch -n 1 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# =========================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/nq_search'

# Ray 临时目录
export RAY_TMPDIR='/data/zmx/SEARCH_R1/tmp_ray'

# HuggingFace 镜像与缓存
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=/data/zmx/SEARCH_R1/huggingface_cache

# WandB
export WAND_PROJECT='Search-R1'

# =========================
# 模型
# =========================
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME="nq-search-r1-alpha-${SEARCH_PENALTY_ALPHA}"

# vLLM 后端
export VLLM_ATTENTION_BACKEND=XFORMERS

# 减少 CUDA 内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    \
    `# =========================` \
    `# Batch` \
    `# =========================` \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    \
    `# =========================` \
    `# 长度大幅缩减` \
    `# =========================` \
    data.max_prompt_length=2048 \
    data.max_response_length=128 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    \
    data.shuffle_train_dataloader=True \
    \
    `# =========================` \
    `# GRPO` \
    `# =========================` \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    \
    `# =========================` \
    `# 模型` \
    `# =========================` \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.torch_dtype=bfloat16 \
    \
    `# =========================` \
    `# Actor Optim` \
    `# =========================` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    \
    `# PPO Batch` \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    \
    `# =========================` \
    `# FSDP` \
    `# =========================` \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    \
    `# =========================` \
    `# Rollout` \
    `# =========================` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.n_agent=2 \
    ++actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.temperature=0.8 \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    \
    `# =========================` \
    `# Ref` \
    `# =========================` \
    ++actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# =========================` \
    `# Trainer` \
    `# =========================` \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    \
    `# 降低 checkpoint/test 频率减少中断` \
    trainer.save_freq=200 \
    trainer.test_freq=100 \
    \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    \
    `# =========================` \
    `# Search` \
    `# =========================` \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=1 \
    +reward.search_penalty_alpha=0.01 \
    +reward.evidence_beta=0.2 \
    \
    2>&1 | tee ${EXPERIMENT_NAME}.log