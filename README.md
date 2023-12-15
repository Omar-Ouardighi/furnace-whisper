# ChitChatPDF

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)


# Chatbot Application

This is a chatbot application built with Python and Streamlit. It uses OpenAI's GPT-3.5-turbo model to generate responses based on the content of uploaded documents.

## Features

- Upload PDF or TXT files
- Vectorize and build a chain of the uploaded document
- Maintain a chat history that can be deleted at any time

## Requirements

- Python 3.7 or later
- Streamlit
- OpenAI API key

## Docker Usage

If you have Docker installed, you can use it to run the application without needing to install Python or the required packages.

1. Build the Docker image:

```bash
docker build -t chatbot-app .
```
2. Run the Docker container:
```
docker run -p 8501:8501 chatbot-app
```
The application will be accessible at http://localhost:8501.


## Installation

Before running the app locally, you need to install the required Python packages. You can do this using `pip` and the `requirements.txt` file included in this repository. Here are the steps to install the packages:

1. First, clone this repository to your local machine using the following command:
```
   git clone git@github.com:Omar-Ouardighi/ChitChatPDF.git
```

2. Navigate to the project directory:
```
  cd your-repository
```

3. Install the required packages using `pip`:
```
  pip install -r requirements.txt
```

4. Add your API-key to .env file 

5.  Once you have installed the required packages, you can run the Streamlit app locally :
```
  streamlit run app.py
```




