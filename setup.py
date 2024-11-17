from setuptools import setup, find_packages

setup(
    name="instructor-test-clients",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "test_clients": ["py.typed"],
    },
    install_requires=[
        "instructor",
        "pydantic",
        "typing_extensions",
    ],
)
