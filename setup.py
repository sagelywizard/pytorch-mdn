import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-mdn",
    version="0.0.2",
    author="Benjamin Bastian",
    description="A mixture density network module for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sagelywizard/pytorch-mdn",
    project_urls={
        "Bug Tracker": "https://github.com/sagelywizard/pytorch-mdn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "mdn"},
    packages=setuptools.find_packages(where="mdn"),
    python_requires=">=3.6",
)
