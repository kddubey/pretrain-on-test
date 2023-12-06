from setuptools import setup, find_packages


requirements_base = [
    "datasets>=2.10.0",
    "pandas>=1.5.3",
    "scikit-learn>=1.2.2",
    "typed-argument-parser>=1.9.0",
    "torch>=1.12.1",
    "tqdm>=4.27.0",
    "transformers>=4.31.0",
]


requirements_meta_analysis = [
    "bambi>=0.13.0",
    "jupyter>=1.0.0",
    "seaborn>=0.13.0",
]


setup(
    name="pretrain-on-test",
    version="0.0.1",
    url="https://github.com/kddubey/pretrain-on-test/",
    python_requires=">=3.9.0",
    install_requires=requirements_base,
    extras_require={"meta": requirements_meta_analysis},
    author_email="kushdubey63@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
