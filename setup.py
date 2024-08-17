from setuptools import setup, find_packages

setup(
    name="cusadi",
    version="0.1.0",
    description="A library for generating and compiling Casadi functions with GPU support.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Se Hwan Jeon",
    author_email="sehwan@mit.edu",
    url="https://github.com/se-hwan/cusadi",
    packages=find_packages(),
    # package_data={'test': ['test.txt']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'casadi',
        'numpy',
        'matplotlib',
        'torch'
    ],
    setup_requires=['setuptools'],
)
