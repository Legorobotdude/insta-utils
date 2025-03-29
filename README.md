# Instagram Data Analyzer

This Python script analyzes data downloaded from your Instagram account export.

## Features

*   **Unfollower Analysis:** Generates an HTML report (`unfollowers.html`) listing accounts that you follow but do not follow you back. Each username in the report links directly to their Instagram profile.

## Prerequisites

*   Python 3
*   Your Instagram data export, unzipped. Specifically, the `followers_1.json` and `following.json` files are needed for the unfollower analysis, expected in the `connections/followers_and_following/` directory relative to the script.

## Usage

Run the script from your terminal, specifying the analysis task you want to perform.

Currently available tasks:

*   `unfollowers`: Analyzes followers and following lists.

**Command:**

```bash
python instagram_analyzer.py <task_name>
```

**Example:**

To generate the unfollowers report:

```bash
python instagram_analyzer.py unfollowers
```

This will:
1.  Read the follower and following data.
2.  Calculate the accounts that don't follow you back.
3.  Create (or overwrite) the `unfollowers.html` file in the same directory as the script.
4.  Attempt to automatically open the `unfollowers.html` file in your default web browser.

## Future Enhancements (Roadmap)

See `roadmap.md` for planned future analysis features, such as content performance analysis. 