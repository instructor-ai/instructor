# Web Crawler with Content Extraction

## Overview

This crawler is designed to navigate through given start URLs, follow links, and extract content from web pages. The highlight of this crawler is its content extraction capability, which employs an AI response model to generate a summary of the content, including titles and tags.

## How Content Extraction Works

### Extraction Process

The crawler fetches web pages and processes each response. It then utilizes the `extract_content` method, which sends the page's text to an AI model. This model is designed to parse the text and return a structured summary with a title, a summary of the main content, and associated tags.

### Handling of Extraction Outcomes

Each extraction attempt returns an instance of `MaybeContentSummary`, which contains either the extraction result or an error flag with a message. This allows for easy differentiation between successful content extraction and failures due to non-extractable content or other issues.

### AI Model Interaction

The `extract_content` method communicates with the AI using an asynchronous call, ensuring the crawler remains efficient and non-blocking. The AI's response is matched against the `MaybeContentSummary` data model to appropriately handle the output.

### Error Management

When the extraction fails, the crawler prints an error message with the specific issue. This approach ensures that the user is informed about non-extractable content or any errors encountered during the extraction process.

## Usage

To use the crawler, simply initialize it with a list of starting URLs and an `httpx.AsyncClient` instance. Call the `crawl` method to start the process. Ensure to handle `httpx.HTTPError` and generic exceptions for robust error management.

For detailed instructions, refer to the comments within the `Crawler` class methods.

## Dependencies
- Python 3.7+
- httpx
- BeautifulSoup4
- pydantic
- asyncio
- logging

Ensure these are installed in your environment to run the crawler successfully.

## Conclusion
With this crawler, users can automate the tedious task of content extraction from web pages. The AI-powered summarization enables high-quality results, making it a valuable tool for data mining and content analysis.
