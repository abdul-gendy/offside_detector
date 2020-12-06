import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="offside_detector",
    version="0.0.1",
    author="Abdulrahman Elgendy",
    author_email="Abdul.Elgendy@outlook.com",
    description="This repository aims to analyze images from football matches to detect whether a play was offside or not",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdul-gendy/offside_detector",
    download_url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires='>=3.6',
)