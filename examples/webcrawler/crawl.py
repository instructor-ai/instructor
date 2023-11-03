import asyncio
import httpx
import logging


from typing import Optional, List
from pydantic import Field
from bs4 import BeautifulSoup

import openai
import instructor

instructor.patch()


class ContentSummary:
    title: str
    summary: str
    tags: str


class MaybeContentSummary:
    """
    MaybeContentSummary is a wrapper around ContentSummary that
    allows us to return an error message if the extraction fails.

    Set the error flag to True if the there is no contact worth
    extracting.
    """

    result: Optional[ContentSummary] = Field(
        default=None, description="The result of extraction if it exists."
    )
    error: Optional[bool] = Field(default=False)
    message: str = Field(default=None)


class Crawler:
    def __init__(self, client: httpx.AsyncClient, start_urls: List[str]):
        self.start_urls = start_urls
        self.client = client
        self.queue = asyncio.Queue()
        self.seen_urls = set()
        self.completed_urls = set()
        self.num_workers = 10

    async def add_to_queue(self, urls: List[str]):
        for url in urls:
            if url not in self.seen_urls:
                await self.queue.put(url)
                self.seen_urls.add(url)

    async def crawl(self):
        await self.add_to_queue(self.start_urls)

        workers = [asyncio.create_task(self.worker(i)) for i in range(self.num_workers)]
        await self.queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def worker(self, worker_id):
        while True:
            url = await self.queue.get()
            try:
                await self.fetch_and_process(url)
            except asyncio.CancelledError:
                break
            finally:
                self.queue.task_done()

    async def fetch_and_process(self, url: str):
        try:
            response = await self.client.get(url)
            await self.process_response(response)
        except httpx.HTTPError as e:
            logging.error(f"HTTP error for {url}: {e}")
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
        finally:
            self.completed_urls.add(url)

    async def process_response(self, response: httpx.Response):
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)]
        await self.add_to_queue(links)
        await self.extract_content(response)

    async def extract_content(self, response: httpx.Response):
        text = response.text
        content: MaybeContentSummary = await openai.ChatCompletion.acreate(
            response_model=MaybeContentSummary,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class crawler that can extract content from any website.",
                },
                {
                    "role": "user",
                    "content": "Extract the title, summary and tags from this website:",
                },
                {"role": "user", "content": text},
            ],
        )
        match content:
            case MaybeContentSummary(error=True, message=message):
                print(f"Error extracting content from {response.url}: {message}")
            case MaybeContentSummary(result=content):
                print(content)
