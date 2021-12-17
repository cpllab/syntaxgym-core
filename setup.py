from setuptools import setup, find_packages


# Source version from package source
import re

version_file = "syntaxgym/__init__.py"
version_match = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]", open(version_file).read(), re.M
)
if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))


setup(
    install_requires=[
        "click~=7.1.2",
        "docker[docker]~=4.4.4",
        "h5py~=3.2.1",
        "lm-zoo==1.3",
        "numpy~=1.20.1; python_version == '3.7'",
        "pandas~=1.2.3",
        "pyparsing~=3.0.6",
        "requests~=2.25.1",
        "spython[singularity]~=0.1.11",
        "tqdm~=4.59.0",
        "urllib3~=1.26.7",
    ],
    name="syntaxgym",
    packages=find_packages(exclude=["test"]),
    scripts=["bin/syntaxgym"],
    version=version_string,
    python_requires=">=3.6",
    license="MIT",
    description="Evaluate neural network language models on syntactic test suites",
    author="Jon Gauthier",
    author_email="jon@gauthiers.net",
    url="https://syntaxgym.org",
    keywords=["language models", "nlp", "ai"],
)
