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

# Writing better python

In this article, we'll walk through some common mistakes people tend to make when writing python code. Then, we'll look at how these considerations should be used when working on larger machine learning training jobs and projects. Lastly, we'll conclude with some suggestions that you'll be able to apply to your project today.

## Common Mistakes

I think the biggest mistakes that people make is that they write code that's a bit too complex. I try to keep these 4 points in mind when writing code 

- Earn the right for an abstraction
- Use standard libraries
- Write simple functions
- Manage the complexity

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

It's important to note here that the two code examples do the same job. But, by using a standard

## Writing Code for Larger Runs

I think there are three main differences that separate machine learning code as compared to traditional software I've written.

1. There is a high possibility of your job failing 
2. Your data will not be able to fit entirely into memory
3. Your jobs might run for a day and a half

This means that when you write 



