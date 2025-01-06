"""Python setup.py for ratmoseq_extract package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("ratmoseq_extract", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="ratmoseq_extract",
    version=read("ratmoseq_extract", "VERSION"),
    description="Awesome ratmoseq_extract created by jackransomlovell",
    url="https://github.com/jackransomlovell/ratmoseq_extract/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="jackransomlovell",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["ratmoseq_extract = ratmoseq_extract.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
