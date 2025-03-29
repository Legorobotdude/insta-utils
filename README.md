# Instagram Data Analyzer

This Python script analyzes data downloaded from your Instagram account export.

## Features

*   **Unfollower Analysis:** Generates an HTML report (`unfollowers.html`) listing accounts that you follow but do not follow you back. Each username links to their profile.
*   **Post Metadata Analysis:** Analyzes your `posts_1.json` file to provide insights on posting frequency, media type distribution, caption lengths, and hashtag usage. Generates plots (`.png` files) in an output directory (default: `plots/`).
*   **Follower Change Tracking:** Compares your current `followers_1.json` against a history file (`follower_history.json` by default) to identify users who have unfollowed or newly followed you since the last check. Updates the history file.

## Prerequisites

*   Python 3
*   Your Instagram data export, unzipped.
    *   For `unfollowers` task: `connections/followers_and_following/followers_1.json` and `connections/followers_and_following/following.json`
    *   For `post_metadata` task: `content/posts_1.json` (adjust path if needed)
    *   For `track_unfollowers` task: `connections/followers_and_following/followers_1.json`
*   Python libraries: Install required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal, specifying the analysis task you want to perform.

Currently available tasks for `instagram_analyzer.py`:

*   `unfollowers`: Analyzes who you follow that doesn't follow back (from standard data export).
*   `track_unfollowers`: Compares current followers to history to find changes (from standard data export).

Currently available tasks for `dashboard_analyzer.py`:

*   `plot_reach`: Plots daily reach trend.
*   `plot_follows`: Plots daily follows trend.
*   `analyze_audience`: Generates plots for audience demographics (age/gender, top locations).
*   `analyze_posts`: Analyzes detailed post performance (ER, top posts, performance by type).
*   `generate_llm_input`: Loads and summarizes Reach, Follows, Audience, and Post Performance data, prompts for user context, and generates a combined text prompt for pasting into an LLM.

**Commands:**

```bash
# For standard export analysis
python instagram_analyzer.py <task_name> [options]

# For dashboard export analysis
python dashboard_analyzer.py <task_name> [options]
```

**Options for `dashboard_analyzer.py`:**
*   `--data-dir <directory>`: Specify directory containing dashboard CSVs (default: `dashboard_export`).
*   `--output-dir <directory>`: Specify output directory for plots (default: `dashboard_plots`).
*   `--post-file <filename>`: Specify filename for detailed post performance CSV (default: auto-detect).

**Examples (`dashboard_analyzer.py`):**

1.  Plot daily reach:
    ```bash
    python dashboard_analyzer.py plot_reach
    ```
2.  Analyze audience demographics:
    ```bash
    python dashboard_analyzer.py analyze_audience
    ```
3.  Analyze detailed post performance:
    ```bash
    python dashboard_analyzer.py analyze_posts
    ```
4.  Generate the combined prompt for LLM analysis:
    ```bash
    python dashboard_analyzer.py generate_llm_input
    ```

**Outputs (`dashboard_analyzer.py`):**
*   Plotting tasks: Save `.png` plots to the output directory.
*   Analysis tasks: Print summaries to the console and may save plots.
*   `generate_llm_input` task: Prints summaries, asks for context, then prints the final text prompt.

## Future Enhancements (Roadmap)

See `roadmap.md` for planned future analysis features. 