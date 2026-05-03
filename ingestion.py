from datetime import datetime
import logging
import requests
import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ArticleFilter(BaseModel):
    category: str = Field(..., description="'factual_news', 'opinion', 'live_blog', or 'other'")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    reason: str = Field(..., description="Short explanation of why it was categorized this way")

def categorize_article(title: str, description: str) -> bool:
    """Uses gpt-4o-mini to quickly filter out opinion pieces. Returns True if factual_news."""
    if not os.getenv("OPENAI_API_KEY"):
        return True # Skip filter in mock mode
        
    client = instructor.from_openai(OpenAI())
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=ArticleFilter,
            messages=[
                {"role": "system", "content": "You are a news editor filtering out opinion pieces and editorials."},
                {"role": "user", "content": f"Title: {title}\nDescription: {description}\nIs this factual news or an opinion/editorial?"}
            ]
        )
        return result.category == "factual_news"
    except Exception as e:
        logger.error(f"Filter error: {e}")
        return True # Default to true on error

def get_daily_topic(category: str):
    """Fetches today's news for a specific category using NewsAPI"""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("No NEWSAPI_KEY found, falling back to dummy data.")
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "category": category,
            "topic": f"Dummy {category} topic",
            "ground_truth": "Dummy ground truth.",
            "prompt": f"Explain this recent event and its significance: Dummy {category} topic"
        }

    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        if response.get("status") != "ok" or not response.get("articles"):
            raise ValueError(f"NewsAPI error: {response.get('message', 'No articles found')}")
        
        for article in response["articles"]:
            title = article.get("title", "")
            desc = article.get("description", "")
            
            # Filter out opinion pieces
            if categorize_article(title, desc):
                topic_data = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "category": category,
                    "topic": title,
                    "ground_truth": f"{desc} {article.get('content', '')}",
                    "prompt": f"Explain this recent event and its significance: {title}"
                }
                logger.info(f"📰 Today's {category.capitalize()} Topic: {title}")
                return topic_data
            else:
                logger.info(f"⏭️ Skipped opinion piece: {title}")
                
        raise ValueError("No factual articles found after filtering.")
    except Exception as e:
        logger.error(f"Failed to fetch news for {category}: {e}")
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "category": category,
            "topic": f"Fallback Topic (Fetch Error) in {category}",
            "ground_truth": "An error occurred fetching the news.",
            "prompt": "What caused the error?"
        }