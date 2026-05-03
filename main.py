import asyncio
import logging
import os
from ingestion import get_daily_topic
from inference import get_model_answers
from evaluator import evaluate_answer, EvaluationScore
from storage import save_evaluation
from config import settings

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ====================== MAIN ORCHESTRATOR ======================
async def main():
    logger.info("🚀 Starting Daily LLM Benchmark Engine (Multi-Category)...")
    
    for category in settings.categories:
        logger.info(f"\n{'='*40}\nProcessing Category: {category.upper()}\n{'='*40}")
        daily_data = get_daily_topic(category)
        
        model_results = await get_model_answers(daily_data["prompt"])
        
        evaluation_results = {}
        model_answers_only = {}
        
        for model_name, result_data in model_results.items():
            answer = result_data["answer"]
            exact_model = result_data["exact_model"]
            model_answers_only[model_name] = answer
            
            if "Error" in str(answer):
                evaluation_results[model_name] = {
                    "exact_model": exact_model,
                    "score": EvaluationScore(
                        factuality_score=0,
                        hallucination_detected=True,
                        overconfidence_score=0.0,
                        evasiveness_score=0,
                        bias_detected=False,
                        bias_type="none",
                        bias_explanation="N/A",
                        explanation="Model failed to respond."
                    )
                }
                continue
                
            score = evaluate_answer(model_name, answer, daily_data["ground_truth"])
            evaluation_results[model_name] = {
                "exact_model": exact_model,
                "score": score
            }
            
            logger.info(f"Results for {model_name} ({exact_model}):")
            logger.info(score.model_dump_json(indent=2))
        
        save_evaluation(daily_data, model_answers_only, evaluation_results)
        
    logger.info("✅ Daily benchmark completed for all categories! Check the dashboard.")

if __name__ == "__main__":
    asyncio.run(main())