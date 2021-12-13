import setuptools
import os
ROOT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
version_file = open(".package-version")
version = version_file.read().strip().split("=")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "fasttopi",
    packages =["fasttopi"],
    version = f"0.{version[1]}",
    author = "Yamen Ajjour",
    author_email = "yajjour@hotmail.com",
    package_dir={   'fasttopi': '.'},
    long_description_content_type = "text/markdown",
    package_data={'': ['config.yaml','models/*.pkl']},
    description = "news categories classifiers for news title",
    long_description=open(os.path.join(ROOT_DIRECTORY, 'README.md')).read(),
    install_requires =["scikit-learn","pandas","fastapi","uvicorn","pyyaml"],

)
