# Instructor Hub

Welcome to instructor hub, the goal of this project is to provide a set of tutorials and examples to help you get started, and allow you to pull in the code you need to get started with `instructor`

Make sure you're using the latest version of `instructor` by running:

```bash
pip install -U instructor
```

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

## CLI Usage

Instructor hub comes with a command line interface (CLI) that allows you to view and interact with the tutorials and examples and allows you to pull in the code you need to get started with the API.

### List Cookbooks

By running `instructor hub list` you can see all the available tutorials and examples. By clickony (doc) you can see the full tutorial back on this website.

```bash
$ instructor hub list --sort
```

| hub_id | slug                          | title                         | n_downloads |
| ------ | ----------------------------- | ----------------------------- | ----------- |
| 2      | multiple_classification (doc) | Multiple Classification Model | 24          |
| 1      | single_classification (doc)   | Single Classification Model   | 2           |

### Searching for Cookbooks

You can search for a tutorial by running `instructor hub list -q <QUERY>`. This will return a list of tutorials that match the query.

```bash
$ instructor hub list -q multi
```

| hub_id | slug                          | title                         | n_downloads |
| ------ | ----------------------------- | ----------------------------- | ----------- |
| 2      | multiple_classification (doc) | Multiple Classification Model | 24          |

### Reading a Cookbook

To read a tutorial, you can run `instructor hub pull --id <hub_id> --page` to see the full tutorial in the terminal. You can use `j,k` to scroll up and down, and `q` to quit. You can also run it without `--page` to print the tutorial to the terminal.

```bash
$ instructor hub pull --id 2 --page
```

### Pulling in Code

You can pull in the code with `--py --output=<filename>` to save the code to a file, or you cal also run it without `--output` to print the code to the terminal.

```bash
$ instructor hub pull --id 2 --py --output=run.py
$ instructor hub pull --id 2 --py > run.py
```

You can run the code instantly if you `|` it to `python`:

```bash
$ instructor hub pull --id 2 --py | python
```

## Call for Contributions

We're looking for a bunch more hub examples, if you have a tutorial or example you'd like to add, please open a pull request in `docs/hub` and we'll review it.

- [ ] Converting the cookbooks to the new format
- [ ] Validator examples
- [ ] Data extraction examples
- [ ] Streaming examples (Iterable and Partial)
- [ ] Batch Parsing examples
- [ ] Query Expansion examples
- [ ] Batch Data Processing examples
- [ ] Batch Data Processing examples with Cache
