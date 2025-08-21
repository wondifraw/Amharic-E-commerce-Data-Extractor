"""
Setup script for Ethiopian Telegram NER System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ethiopian-ner-system",
    version="1.0.0",
    author="Ethiopian NER Team",
    author_email="team@example.com",
    description="Ethiopian Telegram E-commerce NER System with Vendor Analytics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ethiopian-ner-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        "interpretability": [
            "shap>=0.44.1",
            "lime>=0.2.0.1",
        ],
        "nlp": [
            "nltk>=3.8.1",
            "spacy>=3.7.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "ethiopian-ner=scripts.main_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)