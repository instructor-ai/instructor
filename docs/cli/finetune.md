# Using the Command Line Interface
The instructor CLI provides functionalities for managing fine-tuning jobs on OpenAI.

## Creating a Fine-Tuning Job

### View Jobs Options

```sh
$ instructor jobs --help 
                                                                                                               
 Usage: instructor jobs [OPTIONS] COMMAND [ARGS]...                                                            
                                                                                                               
 Monitor and create fine tuning jobs                                                                           
                                                                                                               
╭─ Options ───────────────────────────────────────────────────────────────────────────────╮
│ --help                            Display the help message.                             │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ cancel                    Cancel a fine-tuning job.                                                         │
│ create-from-file          Create a fine-tuning job from a file.                                             │
│ create-from-id            Create a fine-tuning job from an existing ID.                                     │
│ list                      Monitor the status of the most recent fine-tuning jobs.                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Create from File

The create-from-file command uploads and trains a model in a single step.

```sh
❯ instructor jobs create-from-file --help
                                                                                         
 Usage: instructor jobs create-from-file [OPTIONS] FILE                                  
                                                                                         
 Create a fine-tuning job from a file.                                                   
                                                                                         
╭─ Arguments ───────────────────────────────────────────────────────────────────────────╮
│ *    file      TEXT  Path to the file for fine-tuning [default: None] [required]      │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────╮
│ --model                           TEXT     Model to use for fine-tuning               │
│                                            [default: gpt-3.5-turbo]                   │
│ --poll                            INTEGER  Polling interval in seconds [default: 2]   │
│ --n-epochs                        INTEGER  Number of epochs for fine-tuning           │
│ --batch-size                      TEXT     Batch size for fine-tuning                 │
│ --learning-rate-multiplier        TEXT     Learning rate multiplier for fine-tuning   │
│ --validation-file                 TEXT     Path to the validation file                │
│                                            [default: None]                            │
│ --help                                     Show this message and exit.                │
╰───────────────────────────────────────────────────────────────────────────────────────╯
```

#### Usage

```sh
$ instructor jobs create-from-file transformed_data.jsonl --validation_file validation_data.jsonl --n_epochs 3 --batch_size 16 --learning_rate_multiplier 0.5
```

### Create from ID

The create-from-id command uses an uploaded file and trains a model

```sh
❯ instructor jobs create-from-id --help
                                                                                         
 Usage: instructor jobs create-from-id [OPTIONS] ID                                      
                                                                                         
 Create a fine-tuning job from an existing ID.                                           
                                                                                         
╭─ Arguments ───────────────────────────────────────────────────────────────────────────╮
│ *    id      TEXT  ID of the existing fine-tuning job [default: None] [required]      │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────╮
│ --model                           TEXT     Model to use for fine-tuning               │
│                                            [default: gpt-3.5-turbo]                   │
│ --n-epochs                        INTEGER  Number of epochs for fine-tuning           │
│ --batch-size                      TEXT     Batch size for fine-tuning                 │
│ --learning-rate-multiplier        TEXT     Learning rate multiplier for fine-tuning   │
│ --validation-file-id              TEXT     ID of the uploaded validation file         │
│                                            [default: None]                            │
│ --help                                     Show this message and exit.                │
╰───────────────────────────────────────────────────────────────────────────────────────╯
```

#### Usage

```sh
$ instructor files upload transformed_data.jsonl
$ instructor files upload validation_data.jsonl 
$ instructor files list
...
$ instructor jobs create_from_id <file_id> --validation_file <validation_file_id> --n_epochs 3 --batch_size 16 --learning_rate_multiplier 0.5
```


### Viewing Files and Jobs

#### Viewing Jobs

```sh
$ instructor jobs list 

OpenAI Fine Tuning Job Monitoring                                                
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃                ┃              ┃                ┃     Completion ┃                 ┃                ┃        ┃                 ┃
┃ Job ID         ┃ Status       ┃  Creation Time ┃           Time ┃ Model Name      ┃ File ID        ┃ Epochs ┃ Base Model      ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ftjob-PWo6uwk… │ 🚫 cancelled │     2023-08-23 │            N/A │                 │ file-F7lJg6Z4… │ 3      │ gpt-3.5-turbo-… │
│                │              │       23:10:54 │                │                 │                │        │                 │
│ ftjob-1whjva8… │ 🚫 cancelled │     2023-08-23 │            N/A │                 │ file-F7lJg6Z4… │ 3      │ gpt-3.5-turbo-… │
│                │              │       22:47:05 │                │                 │                │        │                 │
│ ftjob-wGoBDld… │ 🚫 cancelled │     2023-08-23 │            N/A │                 │ file-F7lJg6Z4… │ 3      │ gpt-3.5-turbo-… │
│                │              │       22:44:12 │                │                 │                │        │                 │
│ ftjob-yd5aRTc… │ ✅ succeeded │     2023-08-23 │     2023-08-23 │ ft:gpt-3.5-tur… │ file-IQxAUDqX… │ 3      │ gpt-3.5-turbo-… │
│                │              │       14:26:03 │       15:02:29 │                 │                │        │                 │
└────────────────┴──────────────┴────────────────┴────────────────┴─────────────────┴────────────────┴────────┴─────────────────┘
                                    Automatically refreshes every 5 seconds, press Ctrl+C to exit
```


#### Viewing Files

```sh
$ instructor files list 

OpenAI Files                                                      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓                         
┃ File ID                       ┃ Size (bytes) ┃ Creation Time       ┃ Filename ┃ Purpose   ┃                         
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩                         
│ file-0lw2BSNRUlXZXRRu2beCCWjl │       369523 │ 2023-08-23 23:31:57 │ file     │ fine-tune │                         
│ file-IHaUXcMEykmFUp1kt2puCDEq │       369523 │ 2023-08-23 23:09:35 │ file     │ fine-tune │                         
│ file-ja9vRBf0FydEOTolaa3BMqES │       369523 │ 2023-08-23 22:42:29 │ file     │ fine-tune │                         
│ file-F7lJg6Z47CREvmx4kyvyZ6Sn │       369523 │ 2023-08-23 22:42:03 │ file     │ fine-tune │                         
│ file-YUxqZPyJRl5GJCUTw3cNmA46 │       369523 │ 2023-08-23 22:29:10 │ file     │ fine-tune │                         
└───────────────────────────────┴──────────────┴─────────────────────┴──────────┴───────────┘   
```

# Contributions 

We aim to provide a light wrapper around the API rather than offering a complete CLI. Contributions are welcome! Please feel free to make an issue at [jxnl/instructor/issues](https://github.com/jxnl/instructor/issues) or submit a pull request.

