# Streamlit Dashboard Setup

## Files in this archive
- `requirements.txt`: List of all Python dependencies for the dashboard.
- `README.md`: This file, with setup instructions.

## Setup Steps

1. **Install dependencies**  
   In your project directory, run:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API Key**  
   To enable the ChatGPT plugin in your Streamlit app, you need to set your `OPENAI_API_KEY` in Streamlit Secrets:
   - If you're on **Streamlit Cloud**:
     1. Go to your app's **Settings**.
     2. Select **Secrets**.
     3. Add a new secret with:
        - **Key:** `OPENAI_API_KEY`
        - **Value:** *Your OpenAI API key* (e.g., `sk-...`)
     4. Save.
   - If you're running locally:
     1. Create a file named `.streamlit/secrets.toml` in your project root.
     2. Add:
        ```toml
        [general]
        OPENAI_API_KEY = "sk-..."
        ```

3. **Run the app**  
   ```bash
   streamlit run app.py
   ```
