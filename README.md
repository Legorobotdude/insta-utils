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

Currently available tasks:

*   `unfollowers`: Analyzes who you follow that doesn't follow back.
*   `post_metadata`: Analyzes post metadata (timestamps, captions, media types, hashtags).
*   `track_unfollowers`: Compares current followers to history to find changes.

**Command:**

```bash
python instagram_analyzer.py <task_name> [options]
```

**Options:**
*   `--output-dir <directory>`: Specify output directory for plots (used by `post_metadata`, default: `plots`).
*   `--history-file <filepath>`: Specify path for follower history JSON (used by `track_unfollowers`, default: `follower_history.json`).

**Examples:**

1.  Generate the report of users you follow who don't follow back:
    ```bash
    python instagram_analyzer.py unfollowers
    ```

2.  Run the post metadata analysis:
    ```bash
    python instagram_analyzer.py post_metadata
    ```

3.  Track follower changes (using default history file):
    ```bash
    python instagram_analyzer.py track_unfollowers
    ```

4.  Track follower changes using a custom history file location:
    ```bash
    python instagram_analyzer.py track_unfollowers --history-file data/my_follower_history.json
    ```

**Outputs:**
*   `unfollowers` task: Creates `unfollowers.html` and attempts to open it.
*   `post_metadata` task: Prints summary statistics and saves `.png` plots to the output directory.
*   `track_unfollowers` task: Prints users who unfollowed/followed since the last check and updates the history JSON file.

## Future Enhancements (Roadmap)

See `roadmap.md` for planned future analysis features, including engagement analysis (likes/comments). 