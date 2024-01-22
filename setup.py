from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "implicit>=0.7.2",
        "joblib>=1.3.2",
        "mpi4py>=3.1.5",
        "optuna>=3.0.6",
        "pyarrow>=14.0.1",
        "pyyaml>=6.0",
        "tqdm>=4.66.1",
    ],
)

__version__ = "0.1.1"

setup(
    name="lova",
    version=__version__,
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/Lova",
    python_requires=">=3.6",
    description="Lova (Long-term Value Algorithm) is a recommendation algorithm framework focused on long-term value matching.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
