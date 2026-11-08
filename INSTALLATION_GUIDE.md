# Installation and Usage Guide

## 1. Prerequisites

Before running the application, install:

- Python 3.10 or later
- Git
- Visual Studio Code
- Required Python packages

## 2. Clone the Repository

```bash
git clone https://github.com/nmeghana92/Steam-Game-Analytics.git
cd Steam-Game-Analytics
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Dataset Configuration

Place the cleaned Steam games dataset in the project root directory.

Expected filename:

`games_march2025_cleaned.csv`

The dataset is not included in the GitHub repository because of its large file size.

## 6. Configure the Groq API

Create a `.env` file in the project root directory and add your own API key:

```env
GROQ_API_KEY=your_api_key_here
```

Never upload your actual API key to GitHub.

## 7. Run the Application

```bash
streamlit run app.py
```

## 8. Application Features

- Steam game recommendations
- AI-generated recommendation explanations
- Exploratory data analysis
- Sentiment classification
- Model evaluation and visualizations

## 9. Troubleshooting

- Verify that all dependencies are installed.
- Check that the dataset filename and location are correct.
- Verify that the Groq API key is valid.
- Ensure the virtual environment is activated.
