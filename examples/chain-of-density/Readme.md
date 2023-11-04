# Introduction

This is a simple example which shows how to perform Chain Of Density summarization using GPT-3.5 and utilise the generated output to fine-tune a 3.5 model for production usage.

## Instructions

1. First, install all of the required dependencies by running the command below. We recommend using a virtual environment to install these so that it does not affect your system installation.


```
pip3 install -r chain_of_density.txt
```

2. Download the dataset using `download.py`. We're using the `griffin/chain_of_density` dataset for this example so no worries if you don't have a dataset of your own. This should generate a new `.csv` file in the folder called `output.csv`

```
python3 download.py
```

3. We now need some examples to fine-tune our `3.5` model on. We provide a existing `.jsonl` file to use or you can generate new ones from the dataset using `finetune.py`

>  Don't forget to set an environment variable `OPENAI_API_KEY` in your shell if you wish to regenerate the examples. You can do so using the command `export OPENAI_API_KEY=<api key> ` We'll use it subsequently down the line for our finetuning step too

4. Now that we have a `.jsonl` file with a bunch of examples, let's now run a simple finetuning job

```
instructor jobs create-from-file summarization.jsonl 
```

Voila! Now you've got a new GPT3.5 model that's capable of summarizing text fine-tuned with Chain Of Density.

TODO: Evaluate the quality of the improved summaries using Spacy's Entity counter ( So we can calculate entity / tokens )