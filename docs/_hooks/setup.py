from setuptools import setup, find_packages

setup(
    name='mkdocs-hide-lines-plugin',
    version='0.1.0',
    description='MkDocs plugin to hide lines in code blocks',
    author='Instructor Team',
    author_email='team@instructor-ai.com',
    packages=find_packages(),
    entry_points={
        'mkdocs.plugins': [
            'hide_lines = hide_lines:HideLinesPlugin',
        ],
    },
    install_requires=[
        'mkdocs>=1.0.4',
    ],
)
