import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
import openai

from bigcodebench.gen.util.openai_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import concurrent_call


CONCURRENT_SAMPLE_MODELS = ("o1-", "o3-", "reasoner", "grok-3-mini-beta")


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, reasoning_effort="medium", parallel_workers: int = 1, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.parallel_workers = parallel_workers
    
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        messages = [make_raw_chat_prompt(
            task_prompt=prompt,
            subset=self.subset,
            split=self.split,
            instruction_prefix=self.instruction_prefix,
            response_prefix=self.response_prefix,
            tokenizer=None,
        ) for prompt in prompts]
        if self.parallel_workers > 1 and len(messages) > 1:
            if num_samples != 1:
                raise ValueError("parallel_workers currently supports num_samples=1")
            return self._codegen_messages_via_concurrency(messages)
        if self._uses_concurrent_sample_fanout() and num_samples > 1:
            return self._codegen_batch_via_concurrency(messages, num_samples)

        return self._codegen_api_batch(messages, num_samples)

    def _uses_concurrent_sample_fanout(self) -> bool:
        return any(
            self.name.startswith(model) or self.name.endswith(model)
            for model in CONCURRENT_SAMPLE_MODELS
        )

    def _make_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=self.base_url
        )

    def _request_outputs(self, client: openai.OpenAI, message: str, num_samples: int) -> list[str]:
        ret = make_auto_request(
            client,
            message=message,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            n=num_samples,
        )
        return [item.message.content for item in ret.choices]

    def _codegen_api_batch(self, messages: List[str], num_samples: int) -> List[str]:
        client = self._make_client()
        
        all_outputs = []
        for message in tqdm(messages):
            all_outputs.append(self._request_outputs(client, message, num_samples))
        return all_outputs

    def _codegen_single_message(self, message: str) -> list[str]:
        return self._request_outputs(self._make_client(), message, 1)

    def _codegen_batch_via_concurrency(self, messages: List[str], num_samples: int) -> List[str]:
        batches = concurrent_call(
            num_samples, self._codegen_api_batch, messages, num_samples=1
        )
        return [[element for sublist in item for element in sublist] for item in zip(*batches)]

    def _codegen_messages_via_concurrency(self, messages: List[str]) -> list[list[str]]:
        ordered_outputs: list[list[str] | None] = [None] * len(messages)
        with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(messages))) as executor:
            future_to_index = {
                executor.submit(self._codegen_single_message, message): index
                for index, message in enumerate(messages)
            }
            for future in tqdm(as_completed(future_to_index), total=len(messages)):
                ordered_outputs[future_to_index[future]] = future.result()
        return [outputs for outputs in ordered_outputs if outputs is not None]

    def is_direct_completion(self) -> bool:
        return False
