import enum
from pydantic import BaseModel
import pytest
import instructor


class Sentiment(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    sentiment: Sentiment


test_data = [
    (
        "I absolutely love this product! It has exceeded all my expectations.",
        Sentiment.POSITIVE,
    ),
    (
        "The service was terrible. I will never use this company again.",
        Sentiment.NEGATIVE,
    ),
    (
        "The movie was okay. It had some good moments but overall it was average.",
        Sentiment.NEUTRAL,
    ),
]


@pytest.mark.parametrize("data", test_data)
def test_sentiment_analysis(data):
    sample_data, expected_sentiment = data

    client = instructor.from_provider("google/gemini-2.0-flash-exp")

    response = client.chat.completions.create(
        response_model=SentimentAnalysis,
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis model. Analyze the sentiment of the given text and provide the sentiment (positive, negative, or neutral).",
            },
            {"role": "user", "content": sample_data},
        ],
    )

    assert response.sentiment == expected_sentiment
