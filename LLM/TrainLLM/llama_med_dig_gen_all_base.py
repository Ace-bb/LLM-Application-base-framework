import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Any, Dict, Optional, Union, Tuple

# # 特殊注入use_reentrant 切换默认值
# from torch.utils.checkpoint import checkpoint as old_checkpoint
# def new_checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
#     return old_checkpoint(
#         function,
#         *args,
#         use_reentrant=use_reentrant,
#         **kwargs,
#     )
# import torch.utils.checkpoint
# torch.utils.checkpoint.checkpoint = new_checkpoint

import torch
from torch import nn
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, IterableDataset, set_caching_enabled
from datasets.iterable_dataset import _BaseExamplesIterable, deepcopy, IterableDataset, Features, DatasetInfo
from datasets.iterable_dataset import _HasNextIterator as HasNextIterator

set_caching_enabled(False)

import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint, PredictionOutput, TrainOutput
from collections import OrderedDict
# import evaluate
from typing import Iterator, List, Optional
import numpy as np

logger = logging.getLogger(__name__)



class MultiSourcesExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterables,
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None
    ):
        self.ex_iterables = ex_iterables
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)


    def __iter__(self):
        iterators = [[HasNextIterator(ex) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        ex_idxs =  [0 for _ in self.ex_iterables]

        indices_iterator = self._give_indice_iterator()

        for i in indices_iterator:

            j = ex_idxs[i]

            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i][j])
                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i][j].hasnext():
                    iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])

            except StopIteration:
                iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])
            
            ex_idxs[i] = (j + 1) % len(iterators[i])


    def shuffle_data_sources(self, generator: np.random.Generator) -> "MultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [[ex.shuffle_data_sources(generator) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        return MultiSourcesExamplesIterable(
            ex_iterables, generator=generator, probabilities=self.probabilities
        )

    def shard_data_sources(self, shard_idx: int) -> "MultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        raise NotImplementedError("Sharding a RandomlyCyclingMultiSourcesExamplesIterable is not implemented")

def mkdir_json_dataset(
    json_data_paths: List[str], 
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    features: Optional[Features] = None,
):
    generator = np.random.default_rng(seed)

    json_datasets = [
        [
            load_dataset(
                "json", 
                data_files=data_path, 
                streaming=True,
                split="train",
                features=features
            )
            for data_path in json_data_path
        ]
        for json_data_path in json_data_paths
    ]

    ex_iterables = [[d._ex_iterable for d in json_dataset] for json_dataset in json_datasets]

    ex_iterable = MultiSourcesExamplesIterable(
        ex_iterables, 
        generator=generator, 
        probabilities=probabilities
    )

    flatten_json_datasets = []
    for item in json_datasets:
        flatten_json_datasets.extend(item)

    info = DatasetInfo.from_merge([d.info for d in flatten_json_datasets])

    token_per_repo_id = {
        repo_id: token for dataset in flatten_json_datasets for repo_id, token in dataset._token_per_repo_id.items()
    }

    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=None, token_per_repo_id=token_per_repo_id)


@dataclass
class MyArguments:
    data_path: str = field(
        default="",
        metadata={
            "help": (
                "data_path"
            )
        },
    )

    max_eval_dataset_size: int = field(
        default=16384, 
        metadata={"help": "max_eval_dataset_size"}
    )

    eval_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to eval"}
    )

    eval_print_gen_example_count: int = field(
        default=8, 
        metadata={"help": "eval_print_gen_example_count"}
    )

    model_name: str = field(
        default="bigscience/bloom-560m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: str = field(
        default="bigscience/bloom",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    adapter_size: int = field(
        default=128, 
        metadata={"help": "adapter_size"}
    )

    rope_alpha: int = field(
        default=1, 
        metadata={"help": "rope_alpha"}
    )

    train_max_len: int = field(
        default=2048, 
        metadata={"help": "train_max_len"}
    )

    gen_max_len: int = field(
        default=512, 
        metadata={"help": "gen_max_len"}
    )

    pretrain_cut_step: int = field(
        default=1536, 
        metadata={"help": "gen_max_len"}
    )


    def __post_init__(self):
        pass


class MyTrainer(Seq2SeqTrainer):


    def __init__(self, my_args: MyArguments, args: Seq2SeqTrainingArguments, **kwargs):

        from transformers import LlamaTokenizer, LlamaForCausalLM

        # 类型注释
        self.train_dataset: IterableDataset
        self.eval_dataset: IterableDataset
        self.args: Seq2SeqTrainingArguments

        tokenizer = LlamaTokenizer.from_pretrained(
            my_args.tokenizer_name, 
            padding_side='right',
            pad_token='<pad>',
            # additional_special_tokens=[
            #     "<|modelname|>",
            #     "<|modelorg|>",
            # ],
        )

        self.my_args = my_args

        def model_init():
            # from modeling_llama import LlamaForCausalLM
            # model
            model = LlamaForCausalLM.from_pretrained(
                my_args.model_name,
                pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
                torch_dtype=torch.bfloat16,
                adapter_size=my_args.adapter_size,
                rope_alpha=my_args.rope_alpha,
            )
            # model = torch.compile(model)

            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = model.config.vocab_size
            logger.warning(("resize_token_embeddings", len(tokenizer), embedding_size))

            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            if my_args.adapter_size > 0:
                # 锁定fp以外其他参数
                for name, p in model.named_parameters():
                    if "adapter_" in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

            if is_deepspeed_zero3_enabled():
                n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters() if p.requires_grad).values())
            else:
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

            logger.info(f"Training new model from scratch - Trainable size={trainable_n_params/2**20:.2f}M params - Total size={n_params/2**20:.2f}M params")

            return model

        dig_features = datasets.Features(
            {
                'instruction': datasets.Value("string"),
                "digs": datasets.Sequence(feature={
                        "speaker": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        'choices': datasets.Sequence(datasets.Value("string")),
                    }
                ),
            }
        )

        # raw_datasets
        raw_train_datasets = mkdir_json_dataset(
            json_data_paths=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                [
                    my_args.data_path + "chatgpt_med_copy-train.jsonl.gz",
                ],
            ], 
            probabilities=[
                # 标准对话训练数据
                1.0
            ],
            seed=args.seed,
            features=dig_features,
        )


        # self.prompt = "Instructions: You are Helper, a large language model trained by ShLab."

        self.speaker_mapper = {
            "from user": "User: ",
            "to user": "Helper: ",
            "to note": "Record: ",
            "to terminal": "Command: ",
            "from terminal": "Terminal: ",
        }

        # 参数
        # ['患者', '医生']
        
        # eval
        # self.rouge = evaluate.load('rouge')
        # self.bleu = evaluate.load('bleu')

        raw_eval_dataset = mkdir_json_dataset(
            json_data_paths=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                [
                    my_args.data_path + "chatgpt_med_copy-validation.jsonl.gz",
                ],

            ], 
            probabilities=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                1.0
            ],
            seed=args.seed,
            features=dig_features,
        )

        return super(MyTrainer, self).__init__(
            args=args,
            model=model_init(), 
            tokenizer=tokenizer,
            train_dataset=raw_train_datasets.shuffle(seed=args.seed, buffer_size=100000),
            eval_dataset=raw_eval_dataset.take(my_args.max_eval_dataset_size),
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest"),
            **kwargs
        )

    def log(self, logs: Dict[str, float]) -> None:
        logger.info(f"trainer | step={self.state.global_step} logs={logs}")
        return super(MyTrainer, self).log(logs=logs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 限制只保存fp
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = OrderedDict(((name, p) for name, p in state_dict.items() if "adapter_" in name))
        return super(MyTrainer, self)._save(output_dir=output_dir, state_dict=state_dict)
    
    def make_train_map(self):
        tokenizer = self.tokenizer
        train_max_len = self.my_args.train_max_len
        pretrain_cut_step = self.my_args.pretrain_cut_step
        # prompt = self.prompt
        speaker_mapper = self.speaker_mapper

        def train_map(batch):

            all_input_ids = []
            all_labels = []

            for prompt, dig in zip(batch['instruction'], batch['digs']):

                input_ids = []
                labels = []

                if len(dig['speaker']) == 0:
                    continue

                # 预训练数据不一样的逻辑
                if "Pretrain" in dig['speaker'][0].lower():

                    for text in dig['text']:
                        input_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
                        labels.extend(tokenizer(text, add_special_tokens=False).input_ids)
                    
                    #文档结束
                    input_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
                    labels.append(tokenizer.convert_tokens_to_ids("</s>"))

                else:
                    prompt_ids = tokenizer("Instructions: " + prompt, add_special_tokens=False).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

                    input_ids += prompt_ids
                    labels += [-100] * len(prompt_ids)

                    for didx, (old_speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                        
                        old_speaker = old_speaker.lower()
                        
                        speaker = None

                        for k,v in speaker_mapper.items():
                            if k in old_speaker:
                                speaker = v
                        
                        if speaker is None:
                            continue

                        # 合并对象
                        dig_ids = tokenizer(speaker + text, add_special_tokens=False).input_ids
                        input_ids += dig_ids
                        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

                        # 生成的
                        if speaker in {"Helper: ", "Record: ", "Command: "}:
                            labels += dig_ids
                            labels += [tokenizer.convert_tokens_to_ids("</s>")]
                        else:
                            labels += [-100] * len(dig_ids)
                            labels += [-100]

                    # input_ids = input_ids[:train_max_len]
                    # labels = labels[:train_max_len]

                # 事先裁剪过长的句子
                if len(input_ids) > train_max_len:
                    for i in range(0, len(input_ids), pretrain_cut_step):
                        cut_input_ids = input_ids[i: i+train_max_len]
                        cut_labels = labels[i: i+train_max_len]

                        # 删除一句回复都没有的情况
                        if len(cut_labels) >= 12 and np.any(np.array(cut_labels)[1:] >= 0):

                            all_input_ids.append(cut_input_ids)
                            all_labels.append(cut_labels)
                else:
                    all_input_ids.append(input_ids)
                    all_labels.append(labels)


            # 统一合并逻辑
            batch_input_ids = [[]]
            batch_labels = [[]]

            for input_ids, labels in zip(all_input_ids, all_labels):

                if len(batch_input_ids[-1]) + len(input_ids) > train_max_len:
                    batch_input_ids.append(input_ids)
                    batch_labels.append(labels)
                else:
                    batch_input_ids[-1].extend(input_ids)
                    batch_labels[-1].extend(labels)

            # 最后一个可能为空
            if len(batch_input_ids[-1]) == 0:
                batch_input_ids.pop(-1)
                batch_labels.pop(-1)

            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
            }

        return train_map

    def train(
        self,
        *args,
        dry_run=False,
        **kwargs,
    ):
        # 加 特殊map 
        old_dataset = self.train_dataset
        self.train_dataset = self.train_dataset.map(
            self.make_train_map(),
            batched=True,
            batch_size=65536,
            # num_proc=32,
            remove_columns=['digs', "instruction"],
            # desc="Running train_map",
        )

        # print(list(self.train_dataset.take(10)))

        if dry_run:
            self.train_dataset = old_dataset
            return TrainOutput(global_step=0, training_loss=0.0, metrics={})

        tmp_result = super(MyTrainer, self).train(*args, **kwargs)
        # 还原
        self.train_dataset = old_dataset
        return tmp_result

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if "use_cache" not in inputs:
            inputs["use_cache"] = False
        
        return super(MyTrainer, self).training_step(model=model, inputs=inputs)


    def make_predict_loss_map(self):
        tokenizer = self.tokenizer
        train_max_len = self.my_args.train_max_len
        pretrain_cut_step = self.my_args.pretrain_cut_step
        # prompt = self.prompt
        speaker_mapper = self.speaker_mapper


        def predict_loss_map(batch):
            batch_input_ids = []
            batch_labels = []

            for prompt, dig in zip(batch['instruction'], batch['digs']):
                if len(dig['speaker']) == 0:
                    continue

                # 预训练数据不一样的逻辑
                if "Pretrain" in dig['speaker'][0].lower():

                    input_ids = []
                    labels = []

                    for text in dig['text']:
                        input_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
                        labels.extend(tokenizer(text, add_special_tokens=False).input_ids)
                    
                    #文档结束
                    input_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
                    labels.append(tokenizer.convert_tokens_to_ids("</s>"))


                else:
                    input_ids = tokenizer("Instructions: " + prompt, add_special_tokens=False).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]
                    labels = [-100] * len(input_ids)

                    for didx, (old_speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                        
                        old_speaker = old_speaker.lower()
                        
                        speaker = None

                        for k,v in speaker_mapper.items():
                            if k in old_speaker:
                                speaker = v
                        
                        if speaker is None:
                            continue

                        # 合并对象
                        dig_ids = tokenizer(speaker + text, add_special_tokens=False).input_ids
                        input_ids += dig_ids
                        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

                        # 生成的
                        if speaker in {"Helper: ", "Record: ", "Command: "}:
                            labels += dig_ids
                            labels += [tokenizer.convert_tokens_to_ids("</s>")]
                        else:
                            labels += [-100] * len(dig_ids)
                            labels += [-100]

                    # input_ids = input_ids[:train_max_len]
                    # labels = labels[:train_max_len]


                for i in range(0, len(input_ids), pretrain_cut_step):
                    cut_input_ids = input_ids[i: i+train_max_len]
                    cut_labels = labels[i: i+train_max_len]

                    # 删除一句回复都没有的情况
                    if len(cut_labels) >= 12 and np.any(np.array(cut_labels)[1:] >= 0):
                        batch_input_ids.append(cut_input_ids)
                        batch_labels.append(cut_labels)


            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
            }

        return predict_loss_map


    # loss预测与train阶段一致（自动均值）
    # test_dataset 必须有dig ，其他随意 自动按照训练集的方式算loss
    def loss_predict(
        self, 
        test_dataset: IterableDataset, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "test",
        dry_run=False,
    ) -> PredictionOutput:

        # 用训练集的方式转化
        deal_test_dataset = test_dataset.map(
            self.make_predict_loss_map(),
            batched=True,
            # num_proc=32,
            remove_columns=['digs', "instruction"],
            # desc="Running " + metric_key_prefix + " - predict_map",
        )
        #调整参数启动loss模式
        self.args.prediction_loss_only = True
        self.args.predict_with_generate = False

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        tmp_predict_output = super(MyTrainer, self).predict(
            test_dataset=deal_test_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix,
        )

        # 只返回metrics
        return PredictionOutput(predictions=tmp_predict_output.metrics[f"{metric_key_prefix}_loss"], label_ids=None, metrics=tmp_predict_output.metrics)

    # 生成映射
    def make_gen_map(self):
        tokenizer = self.tokenizer
        train_max_len = self.my_args.train_max_len
        gen_max_len = self.my_args.gen_max_len
        # prompt = self.prompt
        speaker_mapper = self.speaker_mapper


        def gen_map(batch):
            batch_input_ids = []

            for prompt, dig in zip(batch['instruction'], batch['digs']):

                input_ids = tokenizer("Instructions: " + prompt, add_special_tokens=False).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

                for didx, (old_speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                    old_speaker = old_speaker.lower()
                    
                    speaker = None

                    for k,v in speaker_mapper.items():
                        if k in old_speaker:
                            speaker = v
                    
                    if speaker is None:
                        continue

                    dig_ids = tokenizer(speaker + text, add_special_tokens=False).input_ids
                    input_ids += dig_ids
                    input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

                # 不强制构造开头
                # input_ids += tokenizer("To", add_special_tokens=False).input_ids
                batch_input_ids.append(input_ids[-(train_max_len - gen_max_len):])

            return {
                "input_ids": batch_input_ids,
            }
        
        return gen_map

    # 生成式预测
    # test_dataset 必须有dig 且最后一个对话为患者 ，生成下一句医生的回复
    def gen_predict(
        self, 
        test_dataset: IterableDataset, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "test",
        dry_run=False,
        **gen_kwargs
    ) -> PredictionOutput:

        deal_test_dataset = test_dataset.map(
            self.make_gen_map(),
            batched=True,
            # num_proc=32,
            remove_columns=['digs', 'answer', "instruction"],
            # desc="Running " + metric_key_prefix + " - predict_map",
        )

        old_per_device_eval_batch_size = self.args.per_device_eval_batch_size
        
        #调整参数启动生成模式
        self.args.prediction_loss_only = False
        self.args.predict_with_generate = True
        self.args.per_device_eval_batch_size = 1

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        old_state = torch.random.get_rng_state()
        torch.manual_seed(self.args.seed)
        tmp_predict_output = super(MyTrainer, self).predict(
            test_dataset=deal_test_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix,
            # gen kargs
            max_length=self.my_args.train_max_len,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("</s>"),
            **gen_kwargs
        )
        #还原
        torch.random.set_rng_state(old_state)
        self.args.per_device_eval_batch_size = old_per_device_eval_batch_size

        tmp_prediction_texts = []

        for tmp_prediction, p_dataset_item in zip(tmp_predict_output.predictions, deal_test_dataset):
            p_dataset_item = p_dataset_item['input_ids']

            start_pos = 0
            while start_pos + len(p_dataset_item) < len(tmp_prediction):
                if np.all(tmp_prediction[start_pos: start_pos + len(p_dataset_item)] == p_dataset_item):
                    break
                start_pos += 1

            # 特殊情况
            if start_pos + len(p_dataset_item) >= len(tmp_prediction):
                tmp_prediction_texts.append("")
                continue

            new_start_pos = start_pos + len(p_dataset_item)
            new_end_pos = new_start_pos

            while new_end_pos < len(tmp_prediction) and tmp_prediction[new_end_pos] != self.tokenizer.convert_tokens_to_ids("</s>"):
                new_end_pos += 1

            tmp_prediction_texts.append(self.tokenizer.decode(tmp_prediction[new_start_pos:new_end_pos]))

        # assert len(tmp_prediction_texts) == len(test_dataset)

        return PredictionOutput(predictions=tmp_prediction_texts, label_ids=None, metrics=tmp_predict_output.metrics)

    # # test
    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

    #     loss, generated_tokens, labels = super(MyTrainer, self).prediction_step(
    #         model=model, 
    #         inputs=inputs, 
    #         prediction_loss_only=prediction_loss_only, 
    #         ignore_keys=ignore_keys
    #     )

    #     logger.info("loss: " + str(loss))

    #     return loss, generated_tokens, labels

    # 把数据集转化成生成的一问一答
    def make_eval_to_gen(self):
        # prompt = self.prompt

        def eval_to_gen(batch):
            new_digs = []
            new_answers = []
            new_instruction = []

            for prompt, dig in zip(batch['instruction'], batch['digs']):
                tmp_speakers = []
                tmp_texts = []

                for didx, (speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                    # # 给患者开头加入prompt
                    # if didx == 0:
                    #     text = prompt + text

                    if "To" in speaker and len(tmp_speakers) > 0:
                        # 倒过来取
                        new_digs.append({
                            "speaker": tmp_speakers.copy(),
                            "text": tmp_texts.copy(),
                        })

                        # 加入 speaker
                        new_answers.append(speaker + text)
                        new_instruction.append(prompt)
                    
                    tmp_speakers.append(speaker)
                    tmp_texts.append(text)


            return {
                "instruction": new_instruction,
                "digs": new_digs,
                "answer": new_answers,
            }
        
        return eval_to_gen
        
    # 用 loss evaluate
    def loss_evaluate(
        self,
        eval_dataset: Optional[IterableDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        dry_run=False,
    ) -> PredictionOutput:

        output = self.loss_predict(
            test_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            dry_run=dry_run,
        )

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        return output

    # 用 gen evaluate
    def gen_evaluate(
        self,
        eval_dataset: Optional[IterableDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        dry_run=False,
    ) -> PredictionOutput:

        eval_and_predict_dataset = eval_dataset.map(
            self.make_eval_to_gen(),
            batched=True,
            # num_proc=32,
            # remove_columns=['id'],
            # desc="Running eval_to_predict",
        )

        output = self.gen_predict(
            test_dataset=eval_and_predict_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            dry_run=dry_run,
        )

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        predict_output = output.predictions
        true_output = [item["answer"] for item in eval_and_predict_dataset]

        # metrics
        rouge_results = self.rouge.compute(predictions=predict_output, references=true_output, tokenizer=lambda x: self.tokenizer(x, add_special_tokens=False).input_ids)
        bleu_results = self.bleu.compute(predictions=predict_output, references=true_output, tokenizer=lambda x: self.tokenizer(x, add_special_tokens=False).input_ids)

        output.metrics[metric_key_prefix + "_bleu"] = bleu_results['bleu']
        output.metrics[metric_key_prefix + "_bleu1"] = bleu_results['precisions'][0]
        output.metrics[metric_key_prefix + "_bleu2"] = bleu_results['precisions'][1]
        output.metrics[metric_key_prefix + "_bleu3"] = bleu_results['precisions'][2]
        output.metrics[metric_key_prefix + "_bleu4"] = bleu_results['precisions'][3]

        output.metrics[metric_key_prefix + "_rouge1"] = rouge_results['rouge1']
        output.metrics[metric_key_prefix + "_rouge2"] = rouge_results['rouge2']
        output.metrics[metric_key_prefix + "_rougeL"] = rouge_results['rougeL']
        output.metrics[metric_key_prefix + "_rougeLsum"] = rouge_results['rougeLsum']

        return output

    def evaluate(
        self,
        eval_dataset: Optional[IterableDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        dry_run=False,
    ) -> Dict[str, float]:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset


        if self.my_args.eval_print_gen_example_count > 0:
            # 打印例子
            print_eval_dataset = eval_dataset.take(self.my_args.eval_print_gen_example_count)
            print_eval_dataset = print_eval_dataset.map(
                self.make_eval_to_gen(),
                batched=True,
                # remove_columns=['id'],
                # desc="Running eval_to_predict",
            )

            print_output = self.gen_predict(
                test_dataset=print_eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                dry_run=dry_run,
            )

            if not dry_run:

                predict_output = print_output.predictions
                true_output = [item["answer"] for item in print_eval_dataset]

                logger.info(f"***** Example *****")
                for predict_item, true_item in zip(predict_output, true_output):
                    logger.info("predict_item: ")
                    logger.info(predict_item)
                    logger.info("true_item: ")
                    logger.info(true_item)
                    logger.info(f"****** ----------- ******")


        if self.my_args.eval_with_generate == True:
            output = self.gen_evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                dry_run=dry_run,
            )
        else:
            output = self.loss_evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                dry_run=dry_run,
            )

        if dry_run:
            return {}

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return output.metrics



def main():

    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    # print(sys.argv)
    for item in sys.argv: print(item)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        tmp_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        tmp_args = parser.parse_args_into_dataclasses()

    my_args: MyArguments = tmp_args[0]
    training_args: Seq2SeqTrainingArguments = tmp_args[1]

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
        ],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.add_handler(
        logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
    )
    transformers.utils.logging.enable_explicit_format()

    # deepspeed logger
    from deepspeed.utils.logging import logger as deepspeed_logger
    deepspeed_logger.setLevel(log_level)
    deepspeed_logger.addHandler(
        logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("CUDA_VISIBLE_DEVICES = " + str(os.environ.get("CUDA_VISIBLE_DEVICES")))

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"My parameters {my_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our Trainer
    trainer = MyTrainer(
        my_args=my_args,
        args=training_args,
    )

    # check dataset
    logger.info("***** check train dataset *****")
    trainer.train(dry_run=True)
    logger.info("***** check evaluate dataset *****")
    trainer.evaluate(dry_run=True)

    # output = trainer.evaluate(
    #     eval_dataset=trainer.eval_dataset.select(range(32))
    # )

    # print(output)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()