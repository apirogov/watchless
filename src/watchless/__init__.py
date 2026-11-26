"""watchless -- fetch a YouTube transcript, optionally summarise it.

    watchless.cli        the command (the `watchless` console script)
    watchless.fetch      transcript and metadata acquisition
    watchless.format     captions to markdown
    watchless.llm        OpenAI-compatible chat transport
    watchless.summarize  prompt and output contract
    watchless.settings   environment variables and defaults
    watchless.termux     Android share-sheet installer
"""

__version__ = "0.1.1"

__all__ = ["__version__"]
