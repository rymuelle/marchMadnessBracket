import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="marchMadnessBracket-rmueller", # Replace with your own username
    version="0.0.1",
    author="Ryan Mueller",
    author_email="rmueller@physics.tamu.edu",
    description="Package to make a march madness bracket.",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/rymuelle/marchMadnessBracket",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
