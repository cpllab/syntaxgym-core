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
        "cached-property==1.5.2; python_version < '3.8'",
        "certifi==2020.12.5",
        "chardet==4.0.0",
        "click==7.1.2",
        "colorama==0.4.4",
        "crayons==0.4.0",
        "docker[docker]==4.4.4",
        "h5py==3.2.1",
        "idna==2.10",
        "lm-zoo==1.3",
        "numpy==1.20.1; python_version == '3.7'",
        "pandas==1.2.3",
        "pyparsing==3.0.6",
        "python-dateutil==2.8.1",
        "pytz==2021.1",
        "requests==2.25.1",
        "semver==2.13.0",
        "six==1.15.0",
        "spython[singularity]==0.1.11",
        "tqdm==4.59.0",
        "urllib3==1.26.7",
        "websocket-client==0.58.0",
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
