import re
import pathlib

from setuptools import setup


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


# Function to read version from __version__.py
def get_version():
    version_file = HERE / "minsearch" / "__version__.py"
    version_text = version_file.read_text()
    # Extract the version number using a regular expression
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_text, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# This call to setup() does all the work
setup(
    name="minsearch",
    version=get_version(),
    description="Minimalistic text search engine that uses sklearn and pandas",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/alexeygrigorev/minsearch",
    author="Alexey Grigorev",
    author_email="alexey@datatalks.club",
    license="WTFPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    packages=["minsearch"],
    include_package_data=True,
    install_requires=["pandas", "scikit-learn"],
)
