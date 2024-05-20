## YAML in Patching

The system now supports a new mode, `MD_YAML`, to handle YAML inputs within markdown. This mode allows the system to parse YAML inputs and convert them into JSON for further processing.

To use this mode, you need to specify it when calling the `patch` function, like so:

```python
client = instructor.patch(client, mode=instructor.Mode.MD_YAML)
```