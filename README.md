# Instagram Data Analyzer

This Python script analyzes data downloaded from your Instagram account export, primarily focusing on data from the Professional Dashboard export.

## Features

*   **Dashboard Data Analysis (`dashboard_analyzer.py`):**
    *   Plots daily trends for Reach, Follows, Visits, Views.
    *   Analyzes and plots audience demographics (Age/Gender, Top Locations).
    *   Analyzes detailed post performance (Engagement Rate, Top Posts, Performance by Type).
    *   Generates a consolidated HTML report embedding all generated plots.
    *   **Direct LLM Analysis:** Summarizes dashboard data, prompts for page context, and sends the combined information to an OpenAI-compatible API (like LM Studio) to generate personalized analysis and recommendations.
*   **Standard Export Analysis (`instagram_analyzer.py`):**
    *   Identifies users you follow who don't follow back.
    *   Tracks follower gains/losses over time.

## Prerequisites

*   Python 3
*   Your Instagram data export(s), unzipped.
    *   `dashboard_analyzer.py` expects CSVs in `dashboard_export/`.
    *   `instagram_analyzer.py` expects JSONs in `connections/followers_and_following/`.
*   Python libraries: `pip install -r requirements.txt`
*   **For LLM Analysis:** An OpenAI-compatible API endpoint running (e.g., LM Studio local server). Note the URL (e.g., `http://localhost:1234/v1`).

## Usage

Run the desired script from your terminal, specifying the task.

```bash
# For dashboard export analysis
python dashboard_analyzer.py <task_name> [options]

# For standard export analysis
python instagram_analyzer.py <task_name> [options]
```

**Key Tasks for `dashboard_analyzer.py`:**

*   `plot_reach`, `plot_follows`, `plot_visits`, `plot_views`: Plot individual daily metrics.
*   `analyze_audience`: Analyze and plot audience demographics.
*   `analyze_posts`: Analyze detailed post performance.
*   `generate_report`: Generate all plots and combine them into an HTML report.
*   `llm_analyze`: **(Recommended for insights)** Perform all analyses, prompt for context, call LLM API, and print the AI-generated analysis.

**Options for `dashboard_analyzer.py`:**
*   `--data-dir <directory>`: Specify dashboard CSV directory (default: `dashboard_export`).
*   `--output-dir <directory>`: Specify plot/report output directory (default: `dashboard_plots`).
*   `--post-file <filename>`: Specify detailed post performance CSV filename (default: auto-detect).
*   `--llm-url <URL>`: Set the LLM API endpoint URL (default: `http://localhost:1234/v1`).
*   `--llm-model <name>`: (Optional) Specify the model name for the API.
*   `--llm-key <key>`: (Optional) Provide an API key if required.

**Example (`dashboard_analyzer.py` - LLM Analysis):**

```bash
# Ensure LM Studio server (or other API endpoint) is running
python dashboard_analyzer.py llm_analyze --llm-url "http://localhost:1234/v1"
```

This will load data, print summaries, ask you for context (niche, goals, etc.), call the specified LLM API, and print the AI's response.

## Future Enhancements (Roadmap)

See `roadmap.md` for planned future analysis features. 