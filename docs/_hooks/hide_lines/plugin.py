from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs.plugins import BasePlugin

class HideLinesPlugin(BasePlugin):
    def on_page_markdown(self, markdown: str, *, page: Page, config: Config, files: Files) -> str:
        """Process the markdown content to hide specified lines.

        Args:
            markdown: The markdown content of the page
            page: The page object
            config: The global configuration object
            files: The files collection

        Returns:
            str: The processed markdown content
        """
        lines = markdown.split('\n')
        result: list[str] = []
        skip_next = False

        for line in lines:
            if skip_next:
                skip_next = False
                continue

            if '# hide_next' in line.lower():
                skip_next = True
                continue

            if not any(marker in line.lower() for marker in ['# hide', '<!-- hide -->']):
                result.append(line)

        return '\n'.join(result)
