import os
from setuptools import find_packages, setup


PACKAGE_DIRNAME = "my_pipeline"
ROOT_DIR = os.path.dirname(__file__)

with open(os.path.join(ROOT_DIR, "README.md")) as readme:
    README = readme.read()

def _line_from_file(filename):
    with open(os.path.join(ROOT_DIR, filename)) as f:
        lines = f.readlines()
        return lines

setup(
    name="jukijuki",
    version="beta",
    author="jukiya",
    include_package_data=True,
    description="This is pipeline for me",
    long_description=README,
    author_email="juki.programming@gmail.com",
    requires=_line_from_file("requirements.txt")
)