---
draft: False
date: 2023-11-26
slug: python-generators
tags:
  - generators
  - streaming
  - python
authors:
  - jxnl
  - anmol
---

# Intro to Generators and LLM Streaming in Python

In production apps, latency is crucial, especially in eCommerce or newer chat applications like ChatGPT. Streaming is the solution that enables us to enhance the user experience without the need for faster response times. In this post, we'll explore Python generators from the ground up and then delve into integrating them with LLM streaming using the Instructor library.

## Python Generators: An Efficient Approach to Iterables

Generators in Python are a game-changer for handling large data sets and stream processing. They allow functions to yield values one at a time, pausing and resuming their state, which is a more memory-efficient approach compared to traditional collections that store all elements in memory.

### The Basics: Yielding Values

A generator function in Python uses the `yield` keyword. It yields values one at a time, allowing the function to pause and resume its state.

```python
def count_to_3():
    yield 1
    yield 2
    yield 3

for num in count_to_3():
    print(num)
```
```
1
2
3
```

### Advantages Over Traditional Collections

- **Memory Efficiency**: Only one item is in memory at a time.
- **Lazy Evaluation**: Values are generated on-the-fly.
- **Maintain State**: Automatically maintains state between executions.

### Generator Expressions: A Shortcut

Python also allows creating generators in a single line of code, known as generator expressions. They are syntactically similar to list comprehensions but use parentheses.

```python
squares = (x*x for x in range(10))
```

### Use Cases in Real-World Applications

Generators shine in scenarios like reading large files, data streaming, and pipeline creation for data processing.

## Integrating Generators with LLM Streaming

If you've used ChatGPT, you'll see that the tokens are streamed out one by one, instead of the full response being shown at the end (can you imagine waiting for the full response??). This is made possible by generators. 

Here's how a vanilla openai generator looks:

```python
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

response_generator = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': "What are some good reasons to smile?"}
    ],
    temperature=0,
    stream=True
)

for chunk in response_generator:
    print(chunk, end="")
```

This is great, but what if we want to do some structured extraction on this stream? For instance, we might want to render frontend components that are streamed out by an LLM. The LLM output could be an ordered list of product recommendations on an ecommerce store.

Should we wait for the entire stream to finish before extracting the list of components or can we extract & validate the components in real time? 

In e-commerce, every millisecond matters so the time-to-first-render can differentiate a successful and not-so-successful e commerce store (and i know how a failing e commerce store feels :/ ).


Let's use Instructor to see how we might tackle this!


### Enhanced Personalized E-commerce Product Recommendations

#### Scenario Setup

Imagine an e-commerce platform where the customer profile includes a detailed history of purchases, browsing behavior, product ratings, preferences in various categories, search history, and even responses to previous recommendations. This extensive data is crucial for generating highly personalized and relevant product suggestions.

#### Stream Processing Approach with Extensive Profile Data

- **Process**: The LLM is fed with segmented parts of the extensive customer profile, and product recommendations are streamed back as they are generated, adapting to the nuances of the customer's behavior and preferences.

- **Profile Data Example**:

Let's assume we have the following user profile:

```python
profile_data = """
Customer ID: 12345
Recent Purchases: [Laptop, Wireless Headphones, Smart Watch]
Frequently Browsed Categories: [Electronics, Books, Fitness Equipment]
Product Ratings: {Laptop: 5 stars, Wireless Headphones: 4 stars}
Recent Search History: [best budget laptops 2023, latest sci-fi books, yoga mats]
Preferred Brands: [Apple, AllBirds, Bench]
Responses to Previous Recommendations: {Product123: Interested, Product456: Not Interested}
Loyalty Program Status: Gold Member
Average Monthly Spend: $500
Preferred Shopping Times: Weekend Evenings
...
"""
```
We want to rank the following products for this user:

```python
products = [
    {"product_id": 1, "product_name": "Apple MacBook Air (2023) - Latest model, high performance, portable"},
    {"product_id": 2, "product_name": "Sony WH-1000XM4 Wireless Headphones - Noise-canceling, long battery life"},
    {"product_id": 3, "product_name": "Apple Watch Series 7 - Advanced fitness tracking, seamless integration with Apple ecosystem"},
    {"product_id": 4, "product_name": "Kindle Oasis - Premium e-reader with adjustable warm light"},
    {"product_id": 5, "product_name": "AllBirds Wool Runners - Comfortable, eco-friendly sneakers"},
    {"product_id": 6, "product_name": "Manduka PRO Yoga Mat - High-quality, durable, eco-friendly"},
    {"product_id": 7, "product_name": "Bench Hooded Jacket - Stylish, durable, suitable for outdoor activities"},
    {"product_id": 8, "product_name": "GoPro HERO9 Black - 5K video, waterproof, for action photography"},
    {"product_id": 9, "product_name": "Nespresso Vertuo Next Coffee Machine - Quality coffee, easy to use, compact design"},
    {"product_id": 10, "product_name": "Project Hail Mary by Andy Weir - Latest sci-fi book from a renowned author"}
]
```

Let's now define our models for structured extraction. Note: instructor will conveniently let us use `Iterable` to model an iterable of our class. In this case, once we define our product recommendation model, we can slap on `Iterable` to define what we ultimately want - a (ranked) list of product recommendations. 


```python
import instructor
from openai import OpenAI
from typing import Iterable
from pydantic import BaseModel

client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.JSON)

class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    relevance_score: float

Recommendations = Iterable[ProductRecommendation]
```
Now let's use our instructor patch. Since we don't want to wait for all the tokens to finish, will set stream to `True` and process each product recommendation as it comes in:
```python

prompt = f"Based on the following user profile:\n{profile_data}\nRank the following products from most relevant to least relevant:\n" + '\n'.join(f"{product['product_id']} {product['product_name']}" for product in products)


recommendations_stream = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    response_model=Iterable[ProductRecommendation],
    stream=True,
    messages=[
        {"role": "system", "content": "Generate product recommendations based on the customer profile. Return in order of highest recommended first."},
        {"role": "user", "content": prompt}
    ]
)
```

`recommendations_stream` is a generator! It yields the extracted products as it's processing the stream in real-time. We can simply iterate over this:

```python
for recommendation in recommendations_stream:
    assert isinstance(recommendation, ProductRecommendation)
    print(f"Recommended Product: {recommendation.product_name}")
```


Don't forget to check our [GitHub](https://github.com/jxnl/instructor) for more resources and give us a star if you find the library helpful!

---

If you have any questions or need further clarifications, feel free to reach out or dive into the Instructor library's documentation for more detailed information. Happy coding!