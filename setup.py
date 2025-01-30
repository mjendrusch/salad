import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="salad",
  version="0.0.1",
  author="Michael Jendrusch",
  author_email="michael.jendrusch@embl.de",
  description="protein structure generation with sparse denoising models.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/mjendrusch/salad/",
  packages=setuptools.find_packages(),
  install_requires=[
    "torch>=2.0",
    "alphafold @ git+https://github.com/google-deepmind/alphafold.git@v2.3.1",
    "jax[cuda12]",
    "dm-haiku @ git+https://github.com/deepmind/dm-haiku",
    "numpy",
    "flexloop @ git+https://github.com/mjendrusch/flexloop.git",
    "gemmi"
  ],
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
  ],
)
