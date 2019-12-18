import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spectral_hyperresolution-radekosmulski",
    version="0.0.1",
    author="Radek Osmulski",
    author_email="radek@earthspecies.org",
    description="Implementation of linear reassignment in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earthspecies/spectral_hyperresolution",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
