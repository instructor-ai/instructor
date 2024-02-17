# Instructor Hub

Welcome to instructor hub, the goal of this project is to provide a set of tutorials and examples to help you get started, and allow you to pull in the code you need to get started with `instructor`

## Contributing

We welcome contributions to the instructor hub, if you have a tutorial or example you'd like to add, please open a pull request in `docs/hub` and we'll review it.

1. The code must be in a single file
2. Make sure that its referenced in the `mkdocs.yml`
3. Make sure that the code is unit tested.

### Using pytest_examples

By running the following command you can run the tests and update the examples. This ensures that the examples are always up to date.
Linted correctly and that the examples are working, make sure to include a `if __name__ == "__main__":` block in your code and add some asserts to ensure that the code is working.

```bash
poetry run pytest tests/openai/docs/test_hub.py --update-examples
```

## Command Line Interface

Instructor hub comes with a command line interface (CLI) that allows you to view and interact with the tutorials and examples and allows you to pull in the code you need to get started with the API.

### Listing Available Cookbooks

By running `instructor hub list` you can see all the available tutorials and examples. By clickony (doc) you can see the full tutorial back on this website.

```bash
$ instructor hub list
                         Available Cookbooks
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ hub_id ┃ slug                        ┃ title                       ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      1 │ single_classification (doc) │ Single Classification Model │
└────────┴─────────────────────────────┴─────────────────────────────┘
```

### Viewing a Tutorial

To read a tutorial, you can run `instructor hub show --id <hub_id> --page` to see the full tutorial in the terminal. You can use `j,k` to scroll up and down, and `q` to quit. You can also run it without `--page` to print the tutorial to the terminal.

```bash
$ instructor hub show --id 1 --page
```

### Pulling in Code

To pull in the code for a tutorial, you can run `instructor hub pull --id <hub_id> --py`. This will print the code to the terminal, then you can `|` pipe it to a file to save it.

```bash
$ instructor hub pull --id 1 --py > classification.py
```

## Future Work

This is a experimental in the future we'd like to have some more features like:

- [ ] Options for async/sync code
- [ ] Options for connecting with langsmith
- [ ] Standard directory structure for the code
- [ ] Options for different languages
