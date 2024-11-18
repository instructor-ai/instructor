from mkdocs.plugins import BasePlugin
import re

class HideLinesPlugin(BasePlugin):
    def on_page_markdown(self, markdown, page, config, files):
        """
        The page_markdown event is called after the page markdown is loaded
        from file and can be used to alter the markdown before it is rendered.
        """
        lines = markdown.split("\n")
        result = []
        in_code_block = False
        skip_next = False

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                if not in_code_block:
                    # End of code block
                    result.append(line)
                    continue

            if in_code_block and line.strip().startswith("#hide"):
                skip_next = True
                continue

            if skip_next:
                skip_next = False
                continue

            result.append(line)

        return "\n".join(result)
