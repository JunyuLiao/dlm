# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from .configuration_llada import LLaDAConfig
__all__ = ['LLaDAConfig', 'LLaDAModelLM', 'LLaDAMoeModelLM', 'LLaDA2MoeModelLM']


def __getattr__(name):
    """Load backend-specific model implementations only when requested."""
    if name == 'LLaDAModelLM':
        from .modeling_llada import LLaDAModelLM

        return LLaDAModelLM
    if name == 'LLaDAMoeModelLM':
        from .modeling_fused_olmoe import FusedOlmoeForCausalLM

        return FusedOlmoeForCausalLM
    if name == 'LLaDA2MoeModelLM':
        from .modeling_llada2_moe import LLaDA2MoeModelLM

        return LLaDA2MoeModelLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
