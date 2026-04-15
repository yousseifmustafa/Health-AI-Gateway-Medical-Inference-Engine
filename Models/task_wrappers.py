import asyncio

class TaskWrappersMixin:
    """Specific agentic wrappers that format queries to the underlying models."""

    # Specific Wrappers (Sync)
    def optimize_query(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)

    def generate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.gen_model_name)

    def validate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.val_model_name)

    def summarize(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)

    # Specific Wrappers (Async)
    async def aoptimize_query(self, prompt: str) -> str:
        return await asyncio.to_thread(self.optimize_query, prompt)

    async def agenerate_answer(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate_answer, prompt)

    async def avalidate_answer(self, prompt: str) -> str:
        return await asyncio.to_thread(self.validate_answer, prompt)

    async def asummarize(self, prompt: str) -> str:
        return await asyncio.to_thread(self.summarize, prompt)
