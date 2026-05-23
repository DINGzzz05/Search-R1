# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


def compute_thinking_quality(text: str) -> float:
    """
    启发式评估 <think> 标签内推理过程的质量。
    返回 0~1 之间的分数（0表示无思考或不满足任何规则，1表示全部满足）。
    """
    # 提取所有 <think> 块
    think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if not think_blocks:
        return 0.0

    # 合并所有思考块
    thinking = ' '.join(think_blocks)

    score = 0.0

    # 规则1：证据利用 —— 提到了搜索结果或引用编号
    if re.search(r'Search Result|\[[\d]+\]', thinking):
        score += 0.34

    # 规则2：逻辑结构 —— 包含逻辑连接词（中/英文）
    logic_markers = [
        '因此', '所以', '然而', '但是', '首先', '其次', '然后', '最后',
        'therefore', 'however', 'first', 'second', 'finally', 'because', 'thus'
    ]
    if any(marker in thinking.lower() for marker in logic_markers):
        score += 0.33

    # 规则3：搜索反思 —— 主动计划下一步搜索或承认信息不足
    if re.search(r'need to search|需要搜索|缺乏|not enough info|信息不足|let me search|i need to find', thinking.lower()):
        score += 0.33

    return min(score, 1.0)


# 奖励管理器
class RewardManager():
    """The reward manager.
    """

def __init__(self, tokenizer, num_examine, format_score=0., search_penalty_alpha=0.0,
             evidence_beta=0.0, no_citation_penalty=-0.5, thinking_quality_weight=0.03) -> None:
    self.tokenizer = tokenizer
    self.num_examine = num_examine
    self.format_score = format_score
    self.search_penalty_alpha = search_penalty_alpha
    self.evidence_beta = evidence_beta
    self.no_citation_penalty = no_citation_penalty
    self.thinking_quality_weight = thinking_quality_weight  # 新增的思考质量权重

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 从 meta_info 中提取证据奖励所需的字段
        final_answers = data.meta_info.get('final_answers', [''] * len(data))
        retrieved_docs = data.meta_info.get('retrieved_docs', [{}] * len(data))

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            answer_score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score
            )
            search_count = data_item.non_tensor_batch['search_count_stats']

            # 基础分数：答案得分 + 搜索次数惩罚
            score = answer_score - self.search_penalty_alpha * search_count

            # 思考质量奖励（过程监督）
            thinking_bonus = compute_thinking_quality(sequences_str)
            score += self.thinking_quality_weight * thinking_bonus

            # ---- 证据奖励 ----
            answer_text = final_answers[i]
            docs = retrieved_docs[i]
            if answer_text and docs:   # 确保该轨迹有最终回答和文档记录
                # 导入证据奖励函数（放在文件顶部也可以）
                from path.to.evidence import compute_evidence_reward
                ev_score, no_cite_pen = compute_evidence_reward(
                    answer_text,
                    docs,
                    no_citation_penalty=self.no_citation_penalty
                )
                score += self.evidence_beta * ev_score + no_cite_pen
            # -------------------

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, search_penalty_alpha=config.reward.search_penalty_alpha)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
