---
comments: true
description: Learn how to use Instructor to analyze restaurant call transcripts, extract structured question-answer pairs, and improve your AI call systems with data-driven insights.
tags:
- Call Transcripts
- Data Analysis
- Instructor
- Pydantic
- Structured Data
title: Understanding Call Transcripts with Instructor
date: 2025-04-13
---

# Understanding Call Transcripts with Instructor

Call transcripts contain valuable insights that often remain untapped. Using Instructor, you can extract structured data from restaurant call logs, focusing on question-answer pairs and customer satisfaction metrics to identify improvement opportunities for your AI call systems.

!!! note "This approach isn't limited to customer service calls. It can also be applied to chat conversations with your retrieval augmented generation AI system!"

<!-- more -->

## The Problem: Unstructured Call Data

Restaurant businesses receive hundreds of calls daily about orders, menu options, dietary restrictions, and business details. Without proper analysis, these interactions become dark data - collected but not leveraged for business intelligence.

## The Solution: Structured Extraction with Instructor

Using Instructor with OpenAI, we can transform call transcripts into structured, analyzable data:

```python
from instructor import OpenAI
from typing import Iterable, Literal, List, Optional
from pydantic import BaseModel

# Initialize the OpenAI client with Instructor
client = OpenAI()

# Define the structured output schema
class QAPair(BaseModel):
    question_type: Literal["Ordering", "Menu", "Hours", "Location", "Reservation", "Event", "Payment", "Complaint", "Other"]
    question: str
    answer: str
    source: List[str]  # list of utterance IDs or timestamps
    question_answered: bool  # did the staff provide a relevant answer?
    user_satisfied: Optional[bool] = None  # was the customer satisfied with the answer?
```

This Pydantic model defines what we want to extract from each call transcript, categorizing inquiries and providing service quality metrics.

## Processing Call Transcripts

Let's apply this to a sample call transcript:

```python
# Sample call transcript
call_log = [
    {"id": "u1", "speaker": "Customer", "text": "Hi, are you open on Sunday?"},
    {"id": "u2", "speaker": "Staff", "text": "Yes, we're open from 11am to 9pm."},
    {"id": "u3", "speaker": "Customer", "text": "Do you have gluten-free pizza?"},
    {"id": "u4", "speaker": "Staff", "text": "Uhh... I'm not really sure about that, sorry."},
    {"id": "u5", "speaker": "Customer", "text": "Okay... I guess I'll just get a large margherita pizza."},
    {"id": "u6", "speaker": "Staff", "text": "Sure thing. That'll be ready in 20 minutes."}
]

# Prompt template using Jinja
prompt_template = """
You are an expert at analyzing restaurant phone transcripts. Extract structured question-answer pairs between the customer and restaurant staff.

Each pair should include:
- The type of question (from a fixed set of categories)
- The question (cleaned up)
- The answer (cleaned up)
- Source utterance IDs
- Whether the question was answered at all
- Whether the user was satisfied with the response

<citation_instructions>
When citing source IDs, be extremely precise:
- Include ALL utterance IDs that form part of the Q&A exchange
- Always include both the question utterance(s) and answer utterance(s)
- For questions that span multiple turns, include ALL relevant utterances
- For answers that span multiple turns, include ALL response utterances
- NEVER fabricate source IDs that don't exist in the transcript
- If a customer follows up on the same question, include all related turns
</citation_instructions>

<rules>
- Only include real questions and relevant answers
- Split compound questions into separate entries
- Set question_answered = true if staff attempted to respond, even if the answer is vague
- Set user_satisfied = true only if the customer acknowledges positively
- Set user_satisfied = null if there's no clear reaction
</rules>

Question Type Categories:
- "Ordering" – placing, modifying, or canceling orders
- "Menu" – ingredients, dietary info, availability
- "Hours" – opening/closing times
- "Location" – address, directions, parking
- "Reservation" – booking or checking reservations
- "Event" – large parties, catering, private events
- "Payment" – payment methods, pricing, discounts, gift cards
- "Complaint" – service issues, food quality problems, refunds
- "Other" – any other inquiries not covered above

<transcript>
{% for turn in transcript %}
  <turn>
    <id>{{ turn.id }}</id>
    <speaker>{{ turn.speaker }}</speaker>
    <text>{{ turn.text }}</text>
  </turn>
{% endfor %}
</transcript>
"""

# Extract structured data
qa_pairs = client.chat.completions.create(
    messages=[{
        "role": "user", 
        "content": prompt_template
    }],
    response_model=Iterable[QAPair],
    context={"transcript": call_log}
)

# Process the results
for pair in qa_pairs:
    print(f"Question type: {pair.question_type}")
    print(f"Q: {pair.question}")
    print(f"A: {pair.answer}")
    print(f"Answered: {pair.question_answered}")
    print(f"User satisfied: {pair.user_satisfied}")
    print("---")
```

### Benefits of Using Jinja Templating

Jinja templating with XML structures provides:
- Clean separation of prompting logic from application code
- Dynamic data handling for different transcript formats
- Enhanced readability and maintainability
- Consistent data presentation to the LLM

## Analyzing the Extracted Data

With structured data in hand, we can perform analytics:

```python
import pandas as pd
from collections import Counter

# Convert extracted pairs to a DataFrame
qa_df = pd.DataFrame([pair.dict() for pair in qa_pairs])

# Calculate key metrics
question_type_counts = Counter(qa_df['question_type'])
answer_rate = qa_df['question_answered'].mean() * 100
satisfaction_rate = qa_df[qa_df['user_satisfied'].notna()]['user_satisfied'].mean() * 100

# Analyze satisfaction by question type
satisfaction_by_type = qa_df.groupby('question_type')['user_satisfied'].agg(
    ['mean', 'count']
).sort_values('mean', ascending=False)

print(f"Overall answer rate: {answer_rate:.1f}%")
print(f"Overall satisfaction rate: {satisfaction_rate:.1f}%")
print("\nSatisfaction by question type:")
print(satisfaction_by_type)
```

## Sample Analysis Results

Here's what the output might look like:

| Question Type | Satisfaction Rate | Count |
|---------------|------------------|-------|
| Hours         | 95.8%            | 24    |
| Location      | 92.3%            | 13    |
| Reservation   | 88.7%            | 31    |
| Payment       | 87.2%            | 18    |
| Ordering      | 86.5%            | 74    |
| Event         | 83.1%            | 12    |
| Menu          | 78.4%            | 56    |
| Complaint     | 62.1%            | 29    |
| Other         | 84.6%            | 15    |

## Improving Your AI Call System

**Key metrics:**
- Answer rate: 94.2%
- Satisfaction rate: 82.6%
- Most common question type: Ordering (74 instances)
- Lowest satisfaction: Complaint (62.1%)

This analysis might reveal that complaints about "delivery" have only 45% satisfaction while "refund" complaints have 72% satisfaction. With these insights, you can make targeted improvements to your AI system for delivery issues.

## Conclusion

With Instructor and Pydantic models, you can transform call transcripts into structured data that provides actionable insights to improve your AI call system, ensuring higher customer satisfaction and more efficient operations.

## Scaling Up with Batch Processing

For larger datasets, leverage Instructor's batch processing capabilities to analyze thousands of call transcripts efficiently. The Instructor CLI provides tools for managing batch jobs. For more information, see our [Batch CLI documentation](/cli/batch/).

## Future Directions

While this guide demonstrates core functionality, consider these extensions:

1. **Evaluation Systems**: Create pipelines that automatically score response quality for correctness, empathy, and resolution effectiveness.

2. **Review Applications**: Develop Streamlit apps for human reviewers to validate and correct AI-extracted data.

3. **Model Fine-tuning**: Use labeled data to fine-tune custom models for your restaurant's call patterns.

4. **Real-time Analysis**: Implement these techniques during live calls for immediate coaching.

5. **Offline Batch Processing**: Process historical call data to identify long-term trends.

By combining structured data extraction with these techniques, you can build systems that understand and improve your customer interactions over time.
