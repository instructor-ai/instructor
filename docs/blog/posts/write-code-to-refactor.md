---
draft: False
date: 2024-02-23
tags:
  - Clean Code
  - Code Quality
authors:
  - ivanleomk
  - jxnl
---

# Stop shooting yourself in the foot with python

In this article, we'll walk through some common mistakes people tend to make when writing python code. Then, we'll look at how these considerations should be used when working on larger machine learning training jobs and projects. Lastly, we'll conclude with some suggestions that you'll be able to apply to your project today.

## Common Mistakes

I think the biggest mistakes that people make is that they write code that's a bit too complex. I try to keep these 4 points in mind when writing code 

- Earn the right for an abstraction
- Use standard libraries
- Manage the complexity
- Write simple functions

### Earn the right for an abstraction

When working with code bases, it's often very tempting to write out complex abstractions for your projects. You might have started out wanting to get a simple variable but ended up with something like what we see below.

```python
from dataclasses import dataclass

@dataclass
class Result:
  count:int

def get_result(count:int):
  return Result(count=count)

def main():
  //do something
  result = get_result(10)
  print(result.count)
```

To be clear, this isn't necessarily bad code but we don't need a dataclass here in order for us to achieve the result we want. It would have been sufficient to pass the count back as an integer or use a simple dictionary. 

We might however, migrate to using a dataclass when we have this same result used in multiple areas and we want to make sure that we're passing a validated value with known properties. This is useful for ensuring your code is reliable but notice how it's something that we introduce only as our project gets more complicated and we have more complex workflows that share some values.

```python
from functools import partial

# Level 1
result = 2 + 3

# Level 2
def add(x,y):
	return x+y

result = add(2,3)

# Level 3
add = lambda x,y : x + y
add_partial = partial(2)
result = add_partial(3)
```

Here's another example above too - if you just need a single result, it might be useful to just use a simple implementation. There's no need to make things complex too early and too fast.

### Use Standard Libraries

it's important to use standard libraries like `pandas` when you get the chance rather than spend hours rolling your own version of a specific function you implement - I'm guilty of this a lot.

Let's walk through a simple use-case - you've got a bunch of objects with different keys and you want to save it as a csv file which should have all of the common keys as columns.

```
objects = [{"key1":2, "key4":10},{"key3":3},{"key4":4}]
```

A simple way to do this is to generate a set of all the keys in the objects and then use a csv.DictWriter to write to a csv file like what's done below

```python
import csv

# This is a bad example 
data = [{"key1": 2, "key4": 10}, {"key3": 3}, {"key4": 4}]

keys = set()
for obj in data:
    keys.update(obj.keys())

with open("output.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=keys)
    writer.writeheader()
    writer.writerows(data)
```

But, we could also use pandas. 

```python
import pandas as pd

data = [{"key1": 2, "key4": 10}, {"key3": 3}, {"key4": 4}]
df = pd.DataFrame(data)
df.to_csv("output.csv", index=False)
```

This is a toy example here so the differences between the two code examples are minimal. But we can extrapolate this to more complicated procedures - for instance a model training job. By using a standardized library, we're able to prevent a lot of unforseen issues and take advantage of the time and effort people have spent to make their code robust and reliable. Furthermore, a standardized library is often frequently shared amongst different projects, making it easy to scale up the complexity of the codebase down the line.

### Manage Complexity

Let's give a small example - let's say we'd like to write a class that's able to support different providers (eg. Hugging Face, OpenAI ) and run an embedding job on them. An initial draft might look like this

```python
class EmbeddingModel():
  def __init__(self,model_name,model_type):
    if model_type == "HuggingFace":
      self.model_type = "HuggingFace"
      self.model = SentenceTransformer(model_name)
    elif model_type == "OpenAI":
      self.model_type = "OpenAI"
      self.model = OpenAI()
    else:
      raise ValueError(f"Invalid Model Type of {model_type} was provided. Please make sure to only pass Hugging Face or OpenAI as valid values")
```

This class is **very prone to errors** because we're relying on a single string to determine the logic. Imagine how much more complex we'd need to make the control flow when we add more options and complex configurations. 

In order to prevent these complex errors from happening, we can use a `@classmethod` and a `enum` in order to prevent this from happening. A  `classmethod` allows us to instantiate a class with the right configuration so that we don't run into these complex edge cases. Furthermore, by pre-defining the values that a user can pass in, we can also support more complex initialisation logic.

```python
import enum

class Provider(enum.Enum):
    HUGGINGFACE = "HuggingFace"
    OPENAI = "OpenAI"
    COHERE = "Cohere"

class EmbeddingModel:
    def __init__(self,model_name:str,model_type=Provider):
        self.model_name=model_name
        self.model_type =model_type
	
    @classmethod
    def from_openai(cls, model_name: str):
        return cls(model_name, model_type=Provider.OPENAI)
    
    @classmethod
    def from_hf(cls, model_name: str):
        return cls(
            model_name,
            provider=Provider.HUGGINGFACE,
        )
```

Small things like class methods and enums go a long way in ensuring that code that we write is safe and works as intended.

### Write Simple Functions

The functions I write have ideally three characteristics

- They do one thing and their name reflects it
- They are used at the same level of abstraction
- They are parameterized with nice type hints 

#### Do one thing well 

A lot of times, it's very tempting to write a function that does a lot in one step. In the example above, we can see that we're trying to both format a specific chunk of data and then save it into the database.

```python
def format_data(data):
  headers = extract_headers(data)
	data = format_dataframe(data,headers)
	upload_to_database(data)
  return data

def process():
	data = get_data()
	data = format_data(data)
```

There are two main problems here

- We're introducing a strange side effect that `format_data` doesn't reflect
- We're mutating state, which makes it difficult to debug issues when our logic gets more complicated

A better way to write this logic would include splitting up the saving of data and the formatting of data into two separate functions. That way our functions closely reflect the actual actions that are being performed. We'd also try to minimize the mutation of state, as seen below.

```python
def format_data(data):
  headers = extract_headers(data)
	return format_dataframe(data,headers)

def save_data(data):
  upload_to_database(data)

def process():
	data = get_data()
	formatted_data = format_data(data)
  save_to_database(data)
```

#### Be consistent with abstraction

It's much nicer to keep code at the same level of abstraction. In the example above, we can see two kinds of abstraction levels

- `Implementation Detail` : An example of something like this might be a function with a complex if-else logic 
- `Action Detail` : These tend to be functions with verbs as names - eg. `save_data`, `send_request`

```python
HUGGING_FACE=[
  "BAAI/bge-small-en-v1.5",
]

OPENAI = [
  "text-embedding-3-small"
]

def inference(model_name,data):
  if model_name in HUGGING_FACE:
    model = EmbeddingModel.from_hf(model_name)
  elif model_name in OPENAI:
    model = EmbeddingModel.from_openai(model_name)
  else:
    raise ValueError
  
	embeddings = model.infer(data)
```

We're trying to handle the complex initialisation of the model along with the abstract `.infer` method that the `EmbeddingModel` class provides. As a result, the code looks clunky and messy. A better way to write this would be to have a separate function to handle the initialisation of the model itself

```python
HUGGING_FACE=[
  "BAAI/bge-small-en-v1.5",
]

OPENAI = [
  "text-embedding-3-small"
]

def initialise_model(model_name:str):
	if model_name in HUGGING_FACE:
    model = EmbeddingModel.from_hf(model_name)
  elif model_name in OPENAI:
    model = EmbeddingModel.from_openai(model_name)
  else:
    raise ValueError(f"Invalid model name of {model_name} was provided")

def inference(model_name:str,data:dict):
  model = initialise_model(model_name)
	embeddings = model.infer(data)
```

Alternatively, we could take advantage of the `@classmethod` decorator again and define a method on the class which takes in a model_name and sets the relevant parameters.

## Training Models

I think there are four main differences that separate machine learning code as compared to traditional software I've written for other use-cases such as full-stack work or simple scripts

1. There is a high possibility of your job failing 
2. Your data will not be able to fit entirely into memory
3. Your jobs might run for a day and a half
4. You need to be able to experiment quickly with different configurations

This means that when you write code that is training a model itself, there are a different set of considerations that you need to keep in mind as compared to a simple python script. I find it useful to keep these five suggestions in mind when I write my training scripts.

1. Write the entire pipeline first
2. Build in regular checkpoints
3. Use generators
4. Implement logging
5. Provide abstract functions that are decoupled from the implementation

### Get a small pipeline first

When working with longer training jobs, it's tempting to try doing a YOLO run at the start and seeing if things work. However, as your pipeline gets more and more complex, you might not be able to detect errors in your logic easily. A good way to circumvent this is to work first with a smaller model and a smaller dataset to ensure that your entire pipeline works from start to finish. 

I typically work with a training dataset of 1000-2000 values when I'm writing these scripts. My goal is to ideally have something that can be ran in < 60s at most so that I can check for any quick implementation issues. This has caught many different bugs in my codebase because we can iterate and experiment quickly.

### Using Checkpoints

It's not uncommon for models to fail midway through a long training run either due to a timeout issue or insufficient memory capacity inside the CUDA GPU that you're using to train

![Solving the “RuntimeError: CUDA Out of memory” error | by Nitin Kishore |  Medium](https://miro.medium.com/v2/resize:fit:1400/1*enMsxkgJ1eb9XvtWju5V8Q.png)

Therefore, you'll need to implement some form of checkpoints so that if training fails, you can just resume it from an earlier version. This has been a lifesaver in many circumstances and many standard libraries should support it out of the box.

### Use Generators

Generators allow you to get significant performance improvements. Let's take the following two code examples

```python
	
```



## Conclusion

In this article, we've walked through a few actionable tips that you can use in your projects today to help improve the quality of your code base. The key takeaway here though is that you want to make sure that your code is easily understood by other people and that it's easy to refactor the code.

Ultimately, writing code is difficult because you need to balance the speed of implementation and an urge to keep everything DRY. 

