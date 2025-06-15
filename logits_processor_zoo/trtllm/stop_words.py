
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

from typing import List, Optional, Set
from transformers import PreTrainedTokenizer
import torch


class StopWordsLogitsProcessor:
    """
    A logits processor that stops generation when any of the specified stop words are detected.
    The processor works by boosting the EOS token's logits when a stop word is detected.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    stop_words (List[str]): List of stop words that should trigger generation to stop.
    boost_factor (float): How strongly to boost the EOS token when a stop word is detected.
                         Higher values make it more likely to stop immediately.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_words: List[str], boost_factor: float = 100.0):
        self.eos_token = tokenizer.eos_token_id
        self.boost_factor = boost_factor
        
        # Convert stop words to token sequences
        self.stop_word_tokens: List[List[int]] = []
        for word in stop_words:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            self.stop_word_tokens.append(tokens)
            
        # Track which sequences have already triggered a stop
        self.stopped_sequences: Set[int] = set()

    def __call__(self, req_ids_batch: List[int], logits_batch: List[torch.Tensor],
                 ids_batch: List[List[List[int]]], stream_ptr,
                 client_ids_batch: List[Optional[int]]):
        
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            ids_batch = torch.LongTensor(ids_batch).to(logits_batch.device, non_blocking=True)
            
            # For each sequence in the batch
            for i in range(logits_batch.shape[1]):
                if i in self.stopped_sequences:
                    continue
                    
                # Check if any stop word appears at the end of the sequence
                for stop_tokens in self.stop_word_tokens:
                    if len(ids_batch[i]) >= len(stop_tokens):
                        # Get the last N tokens where N is the length of the stop word
                        last_tokens = ids_batch[i, -len(stop_tokens):].tolist()
                        
                        # If we found a stop word, boost the EOS token
                        if last_tokens == stop_tokens:
                            logits_batch[:, i, self.eos_token] += self.boost_factor
                            self.stopped_sequences.add(i)
                            break

        return logits_batch 