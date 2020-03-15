from setuptools import setup

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="pytimize",
    packages=["pytimize"],
    version="0.0.1a",
    description="Python optimization library for mathematical programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Terry Zheng et al.",
    author_email="pytimize@terrytm.com",
    maintainer="Pytimize Developers",
    maintainer_email="pytimize@terrytm.com",
    url="https://pytimize.terrytm.com",
    python_requires=">=3.8",
    zip_safe=False,
    install_requires=["numpy", "matplotlib"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
    ]
)
