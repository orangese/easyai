from setuptools import setup, find_packages

setup(name = "easyai",
      version = "0.0a",
      description = "AI made easy.",
      long_description = open("README.md").read(),
      url = "https://github.com/orangese/easyai",
      author = "Ryan Park",
      author_email = "22parkr@millburn.org",
      license = None,
      python_requires = ">=3.6.0",
      install_requires = ["numpy", "scipy", "keras", "matplotlib", "pandas", "pillow", "requests", "xlrd", "six",
                          "opencv-python"],
      packages = find_packages(),
      extras_require = {"gpu": ["tensorflow-gpu==1.8.0"], "cpu": ["tensorflow"]},
      zip_safe = False)