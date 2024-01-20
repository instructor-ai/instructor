# Running a Local Ollama Model

## Dependencies

- ollama
- litellm
- setuptools

## Instructions

1. Install Ollama by visiting the website [https://ollama.ai/download](https://ollama.ai/download) and selecting the appropriate operating system.

2. Once installed, open the Ollama app, which should be running in your taskbar.

3. Open the terminal and download a model. For example, to download the llama2 model, run the command:

```bash
ollama run llama2
```

4. In your terminal, start your virtual environment and install the 'litellm[proxy]' package using poetry you can run the command:

```bash
poetry add 'litellm[proxy]'
```

5. Next, install setuptools using the command:

```bash
poetry add setuptools
```

6. Lastly, start the litellm server with the command: `litellm --model ollama/llama2`. This will expose the port on your local machine.

```bash
litellm --model ollama/llama2
```

7. Now you can run the completion!
