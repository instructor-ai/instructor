# Use Vision API to detect products and generate advertising copy

This post demonstrates how to use GPT-4 Vision API and the Chat API to automatically generate advertising copy from product images. This method can be useful for marketing and advertising teams, as well as for e-commerce platforms.

The full code is available on [GitHub](https://www.github.com/openai/gpt-4).

## Building the models

### Product

For the `Product` model, we define a class that represents a product extracted from an image and store the name, key features, and description. The product attributes are dynamically determined based on the content of the image.

Note that it is easy to add Validators and other Pydantic features to the model to ensure that the data is valid and consistent.

```python
class Product(BaseModel):
    """
    Represents a product extracted from an image using AI.

    The product attributes are dynamically determined based on the content
    of the image and the AI's interpretation. This class serves as a structured
    representation of the identified product characteristics.
    """

    name: str = Field(
        description="A generic name for the product.", example="Headphones"
    )
    key_features: Optional[list[str]] = Field(
        description="A list of key features of the product that stand out.",
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        default=None,
    )

    # Can be customized and automatically generated
    def generate_prompt(self):
        prompt = f"Product: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        if self.key_features:
            prompt += f"Key Features: {', '.join(self.key_features)}\n"
        return prompt

    def __repr__(self):
        return self.generate_prompt()
```

### Identified Product

We also define a class that represents a list of products identified in the images. We also add an error flag and message to indicate if there was an error in the processing of the image.

```python
class IdentifiedProduct(BaseModel):
    """
    Represents a list of products identified in the images.
    """

    products: Optional[List[Product]] = Field(
        description="A list of products identified by the AI.",
        example=[
            Product(
                name="Headphones",
                description="Wireless headphones with noise cancellation.",
                key_features=["Wireless", "Noise Cancellation"],
            )
        ],
        default=None,
    )

    error: bool = Field(default=False)
    message: Optional[str] = Field(default=None)
```

### Advertising Copy

Finally, the `AdCopy` models stores the output in a structured format with a headline and the text.

```python
class AdCopy(BaseModel):
    """
    Represents a generated ad copy.
    """

    headline: str = Field(
        description="A short, catchy, and memorable headline for the given product. The headline should invoke curiosity and interest in the product.",
    )
    ad_copy: str = Field(
        description="A long-form advertisement copy for the given product. This will be used in campaigns to promote the product with a persuasive message and a call-to-action with the objective of driving sales.",
    )
    name: str = Field(
        description="The name of the product being advertised.",    )

    def __str__(self):
        return f"{self.name}: \n" + "-" * 100 + f"{self.headline}\n{self.ad_copy}"

    def __repr__(self):
        return str(self)
```

## Calling the API

### Product Detection

The `read_images` function uses OpenAI's vision model to process a list of image URLs and identify products in each of them. We utilize the `instructor` library to patch the OpenAI client for this purpose.

```python
def read_images(image_urls: List[str]):
    """
    Given a list of image URLs, identify the products in the images.
    """

    logger.info(f"Identifying products in images... {len(image_urls)} images")

    return client_image.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=IdentifiedProduct,
        max_tokens=1024, # can be changed
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify products using the given images and generate key features for each product.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                ],
            }
        ],
    )
```

This gives us a list of products identified in all the images.

### Generate advertising copy

Then, we can use the `generate_ad_copy` function to generate advertising copy for each of the products identified in the images.

Two clients are defined for the two different models. This is because the `gpt-4-vision-preview` model is not compatible with the `gpt-4-1106-preview` model in terms of their response format.

```python
def generate_ad_copy(product: Product):
    """
    Given a product, generate an ad copy for the product.
    """

    logger.info(f"Generating ad copy for product: {product.name}")

    return client_copy.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=AdCopy,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are an expert marketing assistant for all products. Your task is to generate an advertisement copy for a product using the name, description, and key features.",
            },
            {"role": "user", "content": product.generate_prompt()},
        ],
    )
```

### Putting it all together

Finally, we can put it all together in a single function that takes a list of image URLs and generates advertising copy for the products identified in the images.

## Input file

The input file is currently a list of image URLs, but this trivial to change to any required format.
