from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
   name="Pytimize",
   version="0.0.1",
   description="Python optimization library for mathematical programming.",
   long_description=long_description,
   author="Pytimize Development Team",
   author_email="pytimize@terrytm.com",
   url="https://pytimize.terrytm.com",
   python_requires='>=3.8',
   packages=["pytimize"],
   install_requires=["numpy", "matplotlib"],
   zip_safe=False
)
