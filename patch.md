
## YAML in Patching

The system now supports a new mode, `MD_YAML`, to handle YAML inputs within markdown. This mode allows the system to parse YAML inputs and convert them into JSON for further processing.

To use this mode, you need to specify it when calling the `patch` function, like so:

```python
client = instructor.patch(client, mode=instructor.Mode.MD_YAML)
```

You can then provide YAML inputs within markdown. Here's an example:

```yaml
Order Details:
Customer: Jason
Items:

Name: Apple, Price: 0.50
Name: Bread, Price: 2.00
Name: Milk, Price: 1.50
```

The system will parse this input and convert it into a JSON object for further processing.