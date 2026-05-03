import asyncio
from litellm import acompletion
import logging
import os
from config import settings

logger = logging.getLogger(__name__)

async def get_model_answers(prompt: str):
    """Async concurrent calls to multiple models"""
    async def call_model(model: str):
        logger.info(f"🤖 Asking {model}...")
        
        needs_openai = "gpt" in model and not os.getenv("OPENAI_API_KEY")
        needs_anthropic = "claude" in model and not os.getenv("ANTHROPIC_API_KEY")
        
        if needs_openai or needs_anthropic:
            await asyncio.sleep(1) # simulate network delay
            logger.warning(f"No API key for {model}, returning mock response.")
            return model, {"answer": f"Mock response from {model}", "exact_model": f"{model}-mock-v1"}

        try:
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.max_tokens,
                temperature=0.0
            )
            answer = response.choices[0].message.content
            exact_model = response.model if hasattr(response, "model") else model
            logger.info(f"✅ Received answer from {exact_model}")
            return model, {"answer": answer, "exact_model": exact_model}
        except Exception as e:
            logger.error(f"❌ Error with {model}: {str(e)}")
            return model, {"answer": f"Error: {str(e)}", "exact_model": model}

    results = await asyncio.gather(*(call_model(m) for m in settings.models_to_test))
    return dict(results)