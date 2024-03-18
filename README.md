# MLOps API Documentation

## Overview

This MLOps API provides endpoints for assessing adherence and performance of machine learning models. Utilizing FastAPI, this service enables users to calculate model adherence and volumetric performance through easily accessible HTTP requests. Below is a guide to getting started, installing necessary dependencies, and interacting with the API.

## Installation

To run the MLOps API, you will need Python and some additional packages. Follow these steps to set up your environment:

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps

1. Clone the repository or download the source code to your local machine.

2. Navigate to the project directory in your terminal.

3. Create a virtual environment to manage your project's dependencies separately. You can do this by running:

    ```sh
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

    - On Unix or MacOS:

        ```sh
        source venv/bin/activate
        ```

5. Install the required dependencies by running:

    ```sh
    pip install fastapi uvicorn pandas scikit-learn scipy pydantic
    ```

6. Once the dependencies are installed, you are ready to run the server. From the project directory, start the server using uvicorn:

    ```sh
    uvicorn main:app --reload
    ```

    The `--reload` flag makes the server restart after code changes. This is useful for development but should be omitted in a production environment.

## API Endpoints

The API provides two main functionalities: adherence calculation and performance evaluation.

### Adherence Endpoint

- **Endpoint:** `/v1/aderencia`
- **Method:** POST
- **Description:** Calculates the adherence of provided datasets against a preloaded model.

- **Request Body:**

    ```json
    [
        {
            "path": "path/to/your/dataset.csv"
        }
    ]
    ```

- **Response:**

    ```json
    [
        {
            "path": "path/to/your/dataset.csv",
            "ks_stat": 0.2,
            "p_value": 0.05
        }
    ]
    ```

### Performance Volumetry Endpoint

- **Endpoint:** `/v1/performance-volumetry`
- **Method:** POST
- **Description:** Evaluates the volumetric performance and calculates the area under the ROC curve for the given datasets.

- **Request Body:**

    ```json
    [
        {
            "VAR2": "value",
            "IDADE": 30,
            ...
            "REF_DATE": "2023-01-01",
            "TARGET": 1
        }
    ]
    ```

- **Response:**

    ```json
    {
        "2023-01": 1,
        "valor da área sob a curva ROC": 0.85
    }
    ```

## Running the Server

After installation, run the server with the following command:

```sh
uvicorn main:app --host 0.0.0.0 --port 8001
