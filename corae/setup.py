import setuptools 
#import setup, find_packages

with open("README.md", "r") as txt:
    full_description = txt.read()

setuptools.setup(
    name="corae",
    version="0.0.1",
    author="Abdullah Al Mamun",
    author_email="aamcse@gmail.com",
    description="A feature selection framwork",
    long_description=full_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pwaabdullah/CoRAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['keras'],
    python_requires='>=3',
)
