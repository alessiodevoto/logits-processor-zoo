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
from logits_processor_zoo.utils import enforce_tokens


class ForceLastPhraseLogitsProcessor:
    """
    A logits processor which forces LLMs to use the given phrase before they finalize their answers.
    Most common use cases can be providing references, thanking user with context etc.

    Parameters
    ----------
    phrase (str): The phrase to be generated by LLM before the end of its speech.
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    """
    def __init__(self, phrase: str, tokenizer: Union[PreTrainedTokenizer, str]):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.phrase_tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
        self._reset()
        self.phrase = phrase

    # Mutable logits processor gets cloned for each prompt in a batch in order to prevent updating the same object
    # https://github.com/vllm-project/vllm/blob/19dcc02a72e3ed52e3bf95aae44ea1f40ce42ea0/vllm/sampling_params.py#L537-L550
    def clone(self):
        return ForceLastPhraseLogitsProcessor(self.phrase, self.tokenizer)

    def _reset(self):
        self.index = 0

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:  # new generation
            self._reset()

        if scores.argmax() == self.eos_token_id and self.index == 0:
            scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
            self.index += 1
        elif len(self.phrase_tokens) > self.index > 0:
            scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
            self.index += 1

        return scores
