import enum
from itertools import product
from pydantic import BaseModel
import pytest
import instructor
from instructor.function_calls import Mode
from ..util import models, modes


class Sentiment(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    sentiment: Sentiment


test_data = [
    ("I absolutely love this product! It has exceeded all my expectations.", Sentiment.POSITIVE),
    ("The service was terrible. I will never use this company again.", Sentiment.NEGATIVE),
    ("The movie was okay. It had some good moments but overall it was average.", Sentiment.NEUTRAL),
]


@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
def test_sentiment_analysis(model, data, mode, client):
    sample_data, expected_sentiment = data

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")

    client = instructor.patch(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
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