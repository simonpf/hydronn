from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hydronn",
    version="0.0",
    description="Neural-network precipitation nowcasting for Brazil.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/hydronn",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=["numpy", "xarray", "torch", "quantnn"],
    entry_points = {
        'console_scripts': ['hydronn=hydronn.bin:hydronn'],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    project_urls={
        "Source": "https://github.com/simonpf/hydronn/",
    },
    include_package_data=True,
    package_data={'hydronn': ['files/*']},
)
