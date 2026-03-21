import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()



__version__ = "0.0.2"

REPO_NAME = "pylogtracer"
AUTHOR_USER_NAME = "Karthik.K"
AUTHOR_EMAIL = "karthikkashi98@gmail.com"
SRC_REPO = "pylogtracer"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Provider-agnostic Python log analyzer with LLM classification, ReAct agent Q&A, and smart incident clustering",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)