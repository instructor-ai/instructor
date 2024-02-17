# Instructor Hub

Welcome to instructor hub, the goal of this project is to provide a set of tutorials and examples to help you get started, and allow you to pull in the code you need to get started with the OpenAI API.

Every hub example is unit tested and has a full tutorial to help you understand how to use the instructor library.

## Command Line Interface

Instructor hub comes with a command line interface (CLI) that allows you to view and interact with the tutorials and examples and allows you to pull in the code you need to get started with the API.

### Listing Available Cookbooks

By running `instructor hub list` you can see all the available tutorials and examples. By clickony (doc) you can see the full tutorial back on this website.

```bash
$ instructor hub list
```

### Viewing a Tutorial

To read a tutorial, you can run `instructor hub show --id <hub_id> --page` to see the full tutorial in the terminal. You can use `j,k` to scroll up and down, and `q` to quit. You can also run it without `--page` to print the tutorial to the terminal.

```bash
$ instructor hub show --id 1 --page
```

### Pulling in Code

To pull in the code for a tutorial, you can run `instructor hub pull --id <hub_id> --py`. This will print the code to the terminal, then you can `|` pipe it to a file to save it.

!!! note "pbcopy"

    If you are on a Mac, you can use `pbcopy` to copy the code to your clipboard.

    ```bash
    $ instructor hub pull --id 1 --py | pbcopy
    ```

```bash
$ instructor hub pull --id 1 --py > classification.py
```

## Future Work

This is a experimental in the future we'd like to have some more features like:

- [ ] Options for async/sync code
- [ ] Options for connecting with langsmith
- [ ] Standard directory structure for the code
- [ ] Options for different languages
