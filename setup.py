import os
from setuptools import setup


PACKAGE_DIRNAME = "jukijuki"
ROOT_DIR = os.path.dirname(__file__)

with open(os.path.join(ROOT_DIR, "README.md")) as readme:
    README = readme.read()


def get_version():
    version_filepath = os.path.join(ROOT_DIR, PACKAGE_DIRNAME, "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


def _line_from_file(filename):
    with open(os.path.join(ROOT_DIR, filename)) as f:
        lines = f.readlines()
        return lines


setup(
    name=PACKAGE_DIRNAME,
    version=get_version(),
    author="jukiya",
    include_package_data=True,
    description="This is pipeline for tabular, nlp, image competition",
    long_description=README,
    long_description_content_type='text/markdown',
    author_email="juki.programming@gmail.com",
    install_requires=_line_from_file('requirements.txt'),
    packages=[PACKAGE_DIRNAME]
)