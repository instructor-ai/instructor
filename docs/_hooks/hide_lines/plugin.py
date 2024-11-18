from mkdocs.config import Config  # type: ignore
from mkdocs.structure.files import Files  # type: ignore
from mkdocs.structure.pages import Page  # type: ignore
from mkdocs.plugins import BasePlugin  # type: ignore

class HideLinesPlugin(BasePlugin):
    def on_page_markdown(self, markdown: str, *, _page: Page, _config: Config, _files: Files) -> str:
        """Process the markdown content to hide specified lines.

        Args:
            markdown: The markdown content of the page
            _page: The page object (unused but required by plugin interface)
            _config: The global configuration object (unused but required by plugin interface)
            _files: The files collection (unused but required by plugin interface)

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
