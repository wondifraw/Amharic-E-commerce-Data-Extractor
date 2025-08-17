"""Setup script for Amharic E-commerce Data Extractor."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="amharic-ecommerce-extractor",
    version="1.0.0",
    author="Your Name",
    author_email="wondebdu@gmail.com",
    description="Amharic E-commerce Data Extractor & NER Fine-Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wondifraw/Amharic-E-commerce-Data-Extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "amharic-scraper=src.data_collection.telegram_scraper:main",
            "amharic-trainer=src.models.trainer:main",
            "amharic-scorecard=src.analytics.vendor_scorecard:main",
        ],
    },
)