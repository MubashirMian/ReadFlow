# Semantic Book Recommender

**Project Description**

This project develops a semantic book recommender system that leverages Large Language Models (LLMs) and vector search techniques to provide personalized book recommendations based on user preferences.

**Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)

**Introduction**

This project guides you through building a semantic book recommender system using Python, Hugging face, LangChain, and Gradio. The system processes book descriptions, classifies them, and provides recommendations based on user input.

**Features**

- **Data Preprocessing**:
  - Identifying and handling missing data patterns.
  - Verifying the number of categories in the dataset.
  - Removing short descriptions to enhance data quality.
  - Performing final cleaning steps to prepare data for analysis.

- **LLM Integration and Vector Search**:
  - Introduction to Large Language Models (LLMs) and vector search methodologies.
  - Splitting book descriptions using CharacterTextSplitter.
  - Building a vector database for efficient search operations.
  - Implementing vector search to retrieve relevant book recommendations.

- **Zero-Shot Text Classification**:
  - Understanding zero-shot text classification with LLMs.
  - Exploring LLMs for zero-shot classification available on Hugging Face.
  - Classifying book descriptions to categorize books effectively.
  - Evaluating classifier accuracy to ensure reliable classifications.

- **Sentiment Analysis**:
  - Identifying fine-tuned LLMs for sentiment analysis tasks.
  - Extracting emotions from book descriptions to understand sentiment.

- **User Interface**:
  - Building a Gradio dashboard to facilitate user interaction and book recommendations.

**Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MubashirMian/ReadFlow.git
