# I'll check the GitHub stars for the mentioned Python packages.
# The packages are: NumPy, Pandas, Requests, TensorFlow, PyTorch, Flask, Django, SciPy, Matplotlib, Beautiful Soup, Scikit-learn, Jupyter Notebook.

import requests
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

logging.basicConfig(level=logging.INFO)
# GitHub repository names
repos = {
    "NumPy": "numpy/numpy",
    "Pandas": "pandas-dev/pandas",
    "Requests": "psf/requests",
    "TensorFlow": "tensorflow/tensorflow",
    "PyTorch": "pytorch/pytorch",
    "Flask": "pallets/flask",
    "Django": "django/django",
    "SciPy": "scipy/scipy",
    "Matplotlib": "matplotlib/matplotlib",
    "Scikit-learn": "scikit-learn/scikit-learn",
    "Jupyter Notebook": "jupyter/notebook",
    "FastAPI": "tiangolo/fastapi",
    "Hugging Face Transformers": "huggingface/transformers",
    "Lang Chain": "langchain-ai/langchain",
    "Keras": "keras-team/keras",
    "ansible": "ansible/ansible",
    "scikit learn": "scikit-learn/scikit-learn",
    "Open AI Whisper": "openai/whisper",
    "Scrapy": "scrapy/scrapy",
    "Facebook Llama": "facebookresearch/llama",
    "Black": "psf/black",
    "OpenAI gym": "openai/gym",
    "apache airflow": "apache/airflow",
}

pypi_packages = {
    "NumPy": "numpy",
    "Pandas": "pandas",
    "Requests": "requests",
    "TensorFlow": "tensorflow",
    "PyTorch": "torch",
    "Flask": "Flask",
    "Django": "Django",
    "SciPy": "scipy",
    "Matplotlib": "matplotlib",
    "Beautiful Soup": "beautifulsoup4",  # Note the '4' at the end
    "Scikit-learn": "scikit-learn",
    "Jupyter Notebook": "notebook",
    "FastAPI": "fastapi",
    "Hugging Face Transformers": "transformers",
    "Keras": "keras",
    "Ansible": "ansible",
    "Scrapy": "scrapy",
    "Black": "black",
    "OpenAI Gym": "gym",
    "Apache Airflow": "apache-airflow",
    # The following packages might not have direct PyPI equivalents or the names might vary:
    "Lang Chain": "langchain",  # Placeholder, check official sources for actual package name
    "OpenAI": "openai",  # Placeholder, check official sources for actual package name
    "OpenAI Whisper": "openai-whisper",
}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_data_from_url(url):
    response = requests.get(url)
    logging.info("Status code: %s for url: %s", response.status_code, url)
    response.raise_for_status()
    return response.json()


# Wrapper function to handle exceptions
def safe_get_data(url):
    try:
        return get_data_from_url(url)
    except RetryError as e:
        logging.exception(f"Request failed: for url {url}")
        # Return a default value or None
        return None


# Function to get GitHub stars
def get_github_stars(repo_name):
    url = f"https://api.github.com/repos/{repo_name}"
    response = safe_get_data(url)
    if response is None:
        return "Unavailable"
    elif response.status_code == 200:
        data = response.json()
        stars = data["stargazers_count"]
        logging.info(f"{repo_name} has {stars} stars.")
        return stars
    else:
        logging.error(f"Error: {repo_name} has Status code {response.status_code}")
        return "Unavailable"


# Function to get PyPI downloads
def get_pypi_downloads(package_name):
    url = f"https://pypistats.org/api/packages/{package_name}/recent"
    response = safe_get_data(url)
    if response is None:
        return "Unavailable"
    elif response.status_code == 200:
        data = response.json()
        downloads = data["data"]["last_month"]
        logging.info(f"{package_name} has {downloads} downloads.")
        return downloads
    else:
        logging.error(f"Error: {package_name} has Status code {response.status_code}")


def create_charts(data_dict, title):
    # Creating a horizontal bar chart
    logging.info(f"Creating a horizontal bar chart of {title}")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(data_dict.values()), y=list(data_dict.keys()))
    plt.title(f"Bar Chart of {title}")
    plt.xlabel("Metrics")
    plt.ylabel("Packages")
    plt.savefig(f"{title.lower().replace(' ', '_')}_bar_chart.png")
    logging.info(f"Horizontal bar chart of {title} saved successfully.")

    # Creating a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        data_dict.values(), labels=data_dict.keys(), autopct="%1.1f%%", startangle=140
    )
    plt.title(f"Pie Chart of {title}")
    plt.axis("equal")
    plt.savefig(f"{title.lower().replace(' ', '_')}_pie_chart.png")


def main() -> None:
    # Collecting GitHub stars
    github_stars = {package: get_github_stars(repo) for package, repo in repos.items()}
    logging.info("GitHub Stars: %s", github_stars)
    github_stars = {
        package: stars
        for package, stars in github_stars.items()
        if stars != "Unavailable"
    }
    logging.info("GitHub Stars: with unavailables removed %s", github_stars)
    with open("github_stars.txt", "w") as f:
        f.write(str(github_stars))
    # Collecting PyPI downloads
    pypi_downloads = {
        package: get_pypi_downloads(package.lower()) for package in repos.keys()
    }
    logging.info("PyPI Downloads: %s", pypi_downloads)
    pypi_downloads = {
        package: downloads
        for package, downloads in pypi_downloads.items()
        if downloads != "Unavailable"
    }
    logging.info("PyPI Downloads: with unavailables removed %s", pypi_downloads)
    with open("pypi_downloads.txt", "w") as f:
        f.write(str(pypi_downloads))

    # Creating charts
    create_charts(github_stars, "GitHub Stars")
    create_charts(pypi_downloads, "PyPI Downloads")


if __name__ == "__main__":
    main()
