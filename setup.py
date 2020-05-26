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
        "certifi==2020.4.5.1",
        "chardet==3.0.4",
        "click==7.1.2",
        "colorama==0.4.3",
        "crayons==0.3.0",
        "docker==4.2.0",
        "h5py==2.10.0",
        "idna==2.9",
        "lm-zoo==1.1.1",
        "numpy==1.18.4",
        "pandas==1.0.3",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "pytz==2020.1",
        "requests==2.23.0",
        "semver==2.10.1",
        "six==1.14.0",
        "spython==0.0.80",
        "tqdm==4.46.0",
        "urllib3==1.25.9",
        "websocket-client==0.57.0",
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
