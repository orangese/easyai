from setuptools import setup, find_packages

setup(name = "easyai",
      version = "0.0.1b",
      description = "AI made easy.",
      long_description = open("README.md").read(),
      url = "https://github.com/orangese/easyai",
      author = "Ryan Park",
      author_email = "22parkr@millburn.org",
      license = None,
      python_requires = "<3.7.0",
      install_requires = ["numpy", "scipy", "tensorflow-gpu==1.8.0", "keras", "matplotlib", "pandas", "pillow",
                          "requests", "xlrd", "six"],
      packages = ["easyai"],
      zip_safe = False)