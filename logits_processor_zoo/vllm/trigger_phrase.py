#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Union
import torch
from logits_processor_zoo.utils import text_to_token, enforce_tokens


class TriggerPhraseLogitsProcessor:
    """
    A logits processor which triggers phrases when it encounters a given token.

    Parameters
    ----------
    phrase (str): The phrase to be generated by LLM when it encounters the trigger token.
    trigger_token_phrase (str): One token phrase in string to trigger phrases.
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    trigger_count (int): How many times the phrase will be triggered.
    trigger_after (bool): Whether the phrase is written after the trigger token or instead of the trigger token.
    """
    def __init__(self, phrase: str, trigger_token_phrase: str, tokenizer: Union[PreTrainedTokenizer, str],
                 trigger_count: int = 1, trigger_after: bool = False):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.phrase = phrase
        self.trigger_token_phrase = trigger_token_phrase
        self.trigger_count = trigger_count
        self.trigger_token = text_to_token(self.tokenizer, trigger_token_phrase, last=False)
        self.phrase_tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
        self.initial_trigger_count = trigger_count
        self.trigger_after = trigger_after
        self._reset()

    # Mutable logits processor gets cloned for each prompt in a batch in order to prevent updating the same object
    # https://github.com/vllm-project/vllm/blob/19dcc02a72e3ed52e3bf95aae44ea1f40ce42ea0/vllm/sampling_params.py#L537-L550
    def clone(self):
        return TriggerPhraseLogitsProcessor(self.phrase, self.trigger_token_phrase, self.tokenizer,
                                            self.initial_trigger_count, self.trigger_after)

    def _reset(self):
        self.index = -1
        self.trigger_count = self.initial_trigger_count

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:  # new generation
            self._reset()

        if self.trigger_count <= 0:
            return scores

        if scores.argmax() == self.trigger_token and self.index == -1:
            self.index = 0
            if not self.trigger_after:
                scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
                self.index += 1
        elif len(self.phrase_tokens) > self.index >= 0:
            scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
            self.index += 1

        if len(self.phrase_tokens) == self.index:  # phrase completed, reset for next trigger
            self.index = -1
            self.trigger_count -= 1

        return scores
