from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="tagger",
    version="0.0.1",
    description="Library code for Tagging Exercise",
    url="https://github.com/chrka/tagging-exercise",
    author="Christoffer Karlsson",
    author_email="chrka@mac.com",
    license="MIT",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-learn', 'scikit-multilearn',
                      'nltk', 'fasttext', 'tqdm', 'keras', 'tensorflow'],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)