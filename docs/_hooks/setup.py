from setuptools import setup, find_packages

setup(
    name='hide_lines',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'mkdocs.plugins': [
            'hide_lines = hide_lines.plugin:HideLinesPlugin',
        ]
    },
    install_requires=[
        'mkdocs>=1.0.4'
    ]
)
