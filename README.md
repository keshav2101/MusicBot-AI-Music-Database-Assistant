# MusicBot - AI Music Database Assistant

MusicBot is an intelligent chatbot powered by **Ollama** (Llama 3.2) and **LangGraph**. It interacts with the Chinook music database to answer questions about artists, albums, tracks, and sales, and can generate data visualizations.

## Features
- 🎵 **Natural Language Queries**: Ask detailed questions about the music database.
- 📊 **Data Visualization**: Automatically generates bar charts and pie charts.
- 🧠 **Context Awareness**: Remembers previous parts of the conversation.
- 🛡️ **Robustness**: Handles user typos and complex queries.

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running.
   ```bash
   ollama pull llama3.2
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/keshav2101/MusicBot-AI-Music-Database-Assistant.git
   cd Responsible-AI
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `Chinook.db` is in the project directory. (Included in this repo)

## Usage

1. Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```

2. Run the bot:
   ```bash
   python3 ai.py
   ```

## Example Queries
- "Who are the top 5 artists?"
- "Show me a pie chart of sales by country"
- "Count the number of tracks by Iron Maiden"
