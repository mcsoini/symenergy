import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SymEnergy",
    version="0.1",
    author="SymEnergy contributors listed in AUTHORS",
    author_email="soini@posteo.de",
    description="Symbolic energy system optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcsoini/symenergy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
)
