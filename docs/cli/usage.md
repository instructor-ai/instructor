# Using the OpenAI API Usage CLI

The OpenAI API Usage CLI tool provides functionalities for monitoring your OpenAI API usage, breaking it down by model, date, and cost.

## Monitoring API Usage

### View Usage Options

```sh
$ instructor usage --help

 Usage: instructor usage [OPTIONS] COMMAND [ARGS]...

 Check OpenAI API usage data

╭─ Options ───────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                     │
╰─────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────╮
│ list       Displays OpenAI API usage data for the past N days.  │
╰─────────────────────────────────────────────────────────────────╯
```

### List Usage for Specific Number of Days

To display API usage for the past 3 days, use the following command:

```sh
$ instructor usage list -n 3
```

This will output a table similar to:

```plaintext
                 Usage Summary by Date, Snapshot, and Cost
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Date       ┃ Snapshot ID               ┃ Total Requests ┃ Total Cost ($) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 2023-09-04 │ gpt-4-0613                │             44 │           0.68 │
│ 2023-09-04 │ gpt-3.5-turbo-16k-0613    │            195 │           0.84 │
│ 2023-09-04 │ text-embedding-ada-002-v2 │            276 │           0.00 │
│ 2023-09-04 │ gpt-4-32k-0613            │            328 │          49.45 │
└────────────┴───────────────────────────┴────────────────┴────────────────┘
```

### List Usage for Today

To display the API usage for today, simply run:

```sh
$ instructor usage list
```

# Contributions

We aim to provide a light wrapper around the API rather than offering a complete CLI. Contributions are welcome! Please feel free to make an issue at [jxnl/instructor/issues](https://github.com/jxnl/instructor/issues) or submit a pull request.
