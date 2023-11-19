# Introduction

This section includes a list of notebooks that walk you through some simple concepts in Instructor. We start small and then work our way up to more complex and tricky implementations using the library.

## Overview

Currently we have the following notebooks avaliable 

1. `Introduction` - This is a quick walkthrough some of the benefits of Pydantic and how the Instructor Library integrates nicely with Pydantic with `instructor.patch()`

2. `Tips` - Quick demonstration of how to use enums, `Pydantic` models and structured prompting to get specific output formats

3. `Applications Rag`: Learn how to generate nested models with `Pydantic` by rewriting user queries

4. `Knowledge Graphs`: Dive deep into the use of LLMs to break down complex topics into simple knowledge graphs

5. `Validation` : Learn how to use Pydantic's inbuilt validators to perform more complex validation and checks on the outputs of your functions

6. `Chain Of Density` : Learn how to produce high quality summaries that consistently beat out human-generated ones using `Chain of Density` summarization.



## Installation

We utilise the Graphviz package in this tutorial series. If you don't have it on hand, you should download it. Mac users can do so by running `brew install graphviz` while Linux users can try `sudo apt install graphviz` ( modify to your system specific package manager). Here is a link to their official [documentation](https://graphviz.org/download/)

If you're encountering an error like the following when trying to run graphviz after installing it, just restart the notebook and verify you've got graphviz installed by running `dot -v` in your shell.

```
Command '[PosixPath('dot'), '-Kdot', '-Tsvg']' died with <Signals.SIGKILL: 9>.
```

Here are the steps to start running the notebooks

1. Create a virtual environment

```
python3 -m venv .venv
source .venv .venv/bin/activate
```

2. Install the dependencies

```
pip3 install -r requirements.txt
```

3. Add the virtual environment to Jupyter notebook

```
python -m ipykernel install --user --name=instructor-env
```

4. Add OpenAI API Key into your shell by running the following command. This will be set for as long as the shell is open.

```
export OPENAI_API_KEY=<api key goes here>
```

5. Start Jupyter Notebook

```
jupyter notebook
```