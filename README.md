# README for Python Package Popularity Analyzer

## Overview

This Python script, named `Package Popularity Analyzer`, is designed to fetch and compare the popularity metrics of various Python packages. It retrieves data such as the number of GitHub stars and PyPI downloads for a predefined list of Python packages, then generates visual representations of this data in the form of bar and pie charts. The script uses robust error handling and retry mechanisms to ensure reliable data fetching.

## Features

- **GitHub Stars Fetching:** Retrieves the number of stars for specified Python repositories from GitHub.
- **PyPI Downloads Fetching:** Gathers download statistics for Python packages from PyPI.
- **Retry Mechanism:** Implements a retry strategy for network requests using `tenacity`, enhancing reliability.
- **Data Visualization:** Generates horizontal bar charts and pie charts to visualize the popularity metrics.
- **Error Handling:** Gracefully handles and logs errors to allow uninterrupted execution.

## Dependencies

To run this script, you will need Python 3 and the following packages:
- `requests` - for making HTTP requests.
- `matplotlib` and `seaborn` - for data visualization.
- `tenacity` - for retrying network requests.

You can install these dependencies via pip:
```
poetry install
```

## Usage

1. Ensure all dependencies are installed.
2. Run the script using Python:
   ```
   poetry run python -m most_popular_python_packages.main
   ```
3. The script will output `.png` files for the charts and `.txt` files containing the raw data in the script's directory.

## Configuration

The script analyzes a pre-set list of Python packages. You can modify this list by changing the `repos` and `pypi_packages` dictionaries in the script.

## Output

- **Bar Charts:** Horizontal bar charts showing the number of GitHub stars and PyPI downloads for each package.
- **Pie Charts:** Pie charts representing the relative popularity of the packages based on the collected metrics.
- **Text Files:** Files named `github_stars.txt` and `pypi_downloads.txt` containing the raw data.

## Logging

The script logs informational messages and errors, aiding in debugging and providing insights into the script's execution flow.

## Error Handling

Errors, such as network issues, are logged, and the script attempts to retry the request. If data for a particular package cannot be fetched, it's marked as "Unavailable" and the script continues execution.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on the project's repository.

