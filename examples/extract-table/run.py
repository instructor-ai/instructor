from openai import OpenAI

client = OpenAI()


response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this data accurately as a table in markdown format.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # "url": "https://a.storyblok.com/f/47007/2400x1260/f816b031cb/uk-ireland-in-three-charts_chart_a.png/m/2880x0",
                        # "url": "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
                        # "url": "https://a.storyblok.com/f/47007/4800x2766/1688e25601/230629_attoptinratesmidyear_blog_chart02_v01.png/m/2880x0"
                        "url": "https://a.storyblok.com/f/47007/2400x1260/934d294894/uk-ireland-in-three-charts_chart_b.png/m/2880x0"
                    },
                },
                {
                    "type": "text",
                    "text": """
                        First take a moment to reason about the best set of headers for the tables. 
                        Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
                        Then for each table you identified, write a h2 tag that is a descriptive title of the table. 
                        Then follow up with a short description of the what the data is about. 
                        Lastly, produce the markdown table for each table you identified.
                    """,
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
