from multilora.system.LoRALLM import LoRALLM
from dataclasses import dataclass
import asyncio
from typing import List, Optional
from time import time
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GenerateRequest:
    prompt: str
    adapter_id: str
    future: asyncio.Future


@dataclass
class FineTuneRequest:
    prompts: List[str]
    adapter_name: Optional[str]
    future: asyncio.Future


class Processor:
    def add_generate_request(self, request: GenerateRequest):
        raise NotImplementedError()

    def add_finetune_request(self, request: FineTuneRequest):
        raise NotImplementedError()


class InterleaveProcessor(Processor):
    INFERENCE_BATCH_SIZE = 64
    FINETUNE_BATCH_SIZE = 32
    N_INTERLEAVE = 1
    INFERENCE_DELAY = 1

    def __init__(self, llm: LoRALLM):
        self.llm = llm
        self.generate_queue = asyncio.Queue()
        self.finetune_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)

        asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.loop
        )

    def add_generate_request(self, request: GenerateRequest):
        self.generate_queue.put_nowait(request)

    def add_finetune_request(self, request: FineTuneRequest):
        self.finetune_queue.put_nowait(request)

    async def loop(self):
        while True:
            for _ in range(self.N_INTERLEAVE):
                # Collect generate batch
                generate_batch = []
                first_request = await self.generate_queue.get()
                generate_batch.append(first_request)
                start = time()

                while len(generate_batch) < self.INFERENCE_BATCH_SIZE and time() - start < self.INFERENCE_DELAY:
                    try:
                        request = await asyncio.wait_for(self.generate_queue.get(), self.INFERENCE_DELAY)
                    except TimeoutError:
                        break

                    generate_batch.append(request)
                
                inputs = prepare_batch_for_generation(generate_batch)
                try:
                    results = self.llm.generate(inputs)
                    for request, result in zip(generate_batch, results):
                        request.future.set_result(result)
                except Exception as e:
                    for request, result in zip(generate_batch, results):
                        request.future.set_exception(e)

            # Interleaving with one fine-tune request
            if not self.finetune_queue.empty():
                finetune_request = await self.finetune_queue.get()
                inputs = prepare_for_finetuning(finetune_request)
                try:
                    results = self.llm.finetune(inputs)
                    for request, result in zip(generate_batch, results):
                        request.future.set_result("Done")
                except Exception as e:
                    for request, result in zip(generate_batch, results):
                        request.future.set_exception(e)