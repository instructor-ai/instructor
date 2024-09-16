If you want to learn more about concepts in caching and how to use them in your own projects, check out our [blog](../blog/posts/caching.md) on the topic.

## 1. `functools.cache` for Simple In-Memory Caching

**When to Use**: Ideal for functions with immutable arguments, called repeatedly with the same parameters in small to medium-sized applications. This makes sense when we might be reusing the same data within a single session. or in an application where we don't need to persist the cache between sessions.

```python
import time
import functools
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


@functools.cache
def extract(data) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


start = time.perf_counter()  # (1)
model = extract("Extract jason is 25 years old")
print(f"Time taken: {time.perf_counter() - start}")
#> Time taken: 0.41948329200000023

start = time.perf_counter()
model = extract("Extract jason is 25 years old")  # (2)
print(f"Time taken: {time.perf_counter() - start}")
#> Time taken: 1.4579999998431958e-06
```

1. Using `time.perf_counter()` to measure the time taken to run the function is better than using `time.time()` because it's more accurate and less susceptible to system clock changes.
2. The second time we call `extract`, the result is returned from the cache, and the function is not called.

!!! warning "Changing the Model does not Invalidate the Cache"

    Note that changing the model does not invalidate the cache. This is because the cache key is based on the function's name and arguments, not the model. This means that if we change the model, the cache will still return the old result.

Now we can call `extract` multiple times with the same argument, and the result will be cached in memory for faster access.

**Benefits**: Easy to implement, provides fast access due to in-memory storage, and requires no additional libraries.

??? question "What is a decorator?"

    A decorator is a function that takes another function and extends the behavior of the latter function without explicitly modifying it. In Python, decorators are functions that take a function as an argument and return a closure.

    ```python hl_lines="3-5 9"
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Do something before")  # (1)
            #> Do something before
            result = func(*args, **kwargs)
            print("Do something after")  # (2)
            #> Do something after
            return result

        return wrapper


    @decorator
    def say_hello():
        #> Hello!
        print("Hello!")
        #> Hello!


    say_hello()
    #> "Do something before"
    #> "Hello!"
    #> "Do something after"
    ```

    1. The code is executed before the function is called
    2. The code is executed after the function is called

## 2. `diskcache` for Persistent, Large Data Caching

??? note "Copy Caching Code"

    We'll be using the same `instructor_cache` decorator for both `diskcache` and `redis` caching. You can copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import diskcache

    cache = diskcache.Cache('./my_cache_directory')  # (1)


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):  # (2)
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    1. We create a new `diskcache.Cache` instance to store the cached data. This will create a new directory called `my_cache_directory` in the current working directory.
    2. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic in this example code

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Suitable for applications needing cache persistence between sessions or dealing with large datasets. This is useful when we want to reuse the same data across multiple sessions, or when we need to store large amounts of data!

```python hl_lines="10"
import functools
import inspect
import instructor
import diskcache

from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())
cache = diskcache.Cache('./my_cache_directory')


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation  # (4)
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (
            f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  #  (2)
        )
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type (3)
            return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper


class UserDetail(BaseModel):
    name: str
    age: int


@instructor_cache
def extract(data) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.
3. We use Pydantic's `model_validate_json` to deserialize the cached result into a Pydantic model.
4. We use `inspect.signature` to get the function's return type annotation, which we use to validate the cached result.

**Benefits**: Reduces computation time for heavy data processing, provides disk-based caching for persistence.

## 2. Redis Caching Decorator for Distributed Systems

??? note "Copy Caching Code"

    We'll be using the same `instructor_cache` decorator for both `diskcache` and `redis` caching. You can copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import redis

    cache = redis.Redis("localhost")


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Recommended for distributed systems where multiple processes need to access the cached data, or for applications requiring fast read/write access and handling complex data structures.

```python
import redis
import functools
import inspect
import instructor

from pydantic import BaseModel
from openai import OpenAI

client = instructor.from_openai(OpenAI())
cache = redis.Redis("localhost")


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  # (2)
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper


class UserDetail(BaseModel):
    name: str
    age: int


@instructor_cache
def extract(data) -> UserDetail:
    # Assuming client.chat.completions.create returns a UserDetail instance
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.

**Benefits**: Scalable for large-scale systems, supports fast in-memory data storage and retrieval, and is versatile for various data types.

!!! note "Looking carefully"

    If you look carefully at the code above you'll notice that we're using the same `instructor_cache` decorator as before. The implementation is the same, but we're using a different caching backend!

## 4. Prompt Caching with Anthropic

**When to Use**: If you're using the Anthropic api, you can use the `anthropic.beta.prompt_caching.messages.create` method to cache portions of your prompt. This is useful when we need to make multiple calls that utilise shared context.

??? note "Source Text"

    In the following example, we'll be using a short excerpt from the novel "Pride and Prejudice" by Jane Austen. This text serves as an example of a substantial context that might typically lead to slow response times and high costs when working with language models. You can download it manually [here](https://www.gutenberg.org/cache/epub/1342/pg1342.txt)

    ```
        _Walt Whitman has somewhere a fine and just distinction between “loving
    by allowance” and “loving with personal love.” This distinction applies
    to books as well as to men and women; and in the case of the not very
    numerous authors who are the objects of the personal affection, it
    brings a curious consequence with it. There is much more difference as
    to their best work than in the case of those others who are loved “by
    allowance” by convention, and because it is felt to be the right and
    proper thing to love them. And in the sect--fairly large and yet
    unusually choice--of Austenians or Janites, there would probably be
    found partisans of the claim to primacy of almost every one of the
    novels. To some the delightful freshness and humour of_ Northanger
    Abbey, _its completeness, finish, and_ entrain, _obscure the undoubted
    critical facts that its scale is small, and its scheme, after all, that
    of burlesque or parody, a kind in which the first rank is reached with
    difficulty._ Persuasion, _relatively faint in tone, and not enthralling
    in interest, has devotees who exalt above all the others its exquisite
    delicacy and keeping. The catastrophe of_ Mansfield Park _is admittedly
    theatrical, the hero and heroine are insipid, and the author has almost
    wickedly destroyed all romantic interest by expressly admitting that
    Edmund only took Fanny because Mary shocked him, and that Fanny might
    very likely have taken Crawford if he had been a little more assiduous;
    yet the matchless rehearsal-scenes and the characters of Mrs. Norris and
    others have secured, I believe, a considerable party for it._ Sense and
    Sensibility _has perhaps the fewest out-and-out admirers; but it does
    not want them._
    _I suppose, however, that the majority of at least competent votes
    would, all things considered, be divided between_ Emma _and the present
    book; and perhaps the vulgar verdict (if indeed a fondness for Miss
    Austen be not of itself a patent of exemption from any possible charge
    of vulgarity) would go for_ Emma. _It is the larger, the more varied, the
    more popular; the author had by the time of its composition seen rather
    more of the world, and had improved her general, though not her most
    peculiar and characteristic dialogue; such figures as Miss Bates, as the
    Eltons, cannot but unite the suffrages of everybody. On the other hand,
    I, for my part, declare for_ Pride and Prejudice _unhesitatingly. It
    seems to me the most perfect, the most characteristic, the most
    eminently quintessential of its author’s works; and for this contention
    in such narrow space as is permitted to me, I propose here to show
    cause._
    _In the first place, the book (it may be barely necessary to remind the
    reader) was in its first shape written very early, somewhere about 1796,
    when Miss Austen was barely twenty-one; though it was revised and
    finished at Chawton some fifteen years later, and was not published till
    1813, only four years before her death. I do not know whether, in this
    combination of the fresh and vigorous projection of youth, and the
    critical revision of middle life, there may be traced the distinct
    superiority in point of construction, which, as it seems to me, it
    possesses over all the others. The plot, though not elaborate, is almost
    regular enough for Fielding; hardly a character, hardly an incident
    could be retrenched without loss to the story. The elopement of Lydia
    and Wickham is not, like that of Crawford and Mrs. Rushworth, a_ coup de
    théâtre; _it connects itself in the strictest way with the course of the
    story earlier, and brings about the denouement with complete propriety.
    All the minor passages--the loves of Jane and Bingley, the advent of Mr.
    Collins, the visit to Hunsford, the Derbyshire tour--fit in after the
    same unostentatious, but masterly fashion. There is no attempt at the
    hide-and-seek, in-and-out business, which in the transactions between
    Frank Churchill and Jane Fairfax contributes no doubt a good deal to the
    intrigue of_ Emma, _but contributes it in a fashion which I do not think
    the best feature of that otherwise admirable book. Although Miss Austen
    always liked something of the misunderstanding kind, which afforded her
    opportunities for the display of the peculiar and incomparable talent to
    be noticed presently, she has been satisfied here with the perfectly
    natural occasions provided by the false account of Darcy’s conduct given
    by Wickham, and by the awkwardness (arising with equal naturalness) from
    the gradual transformation of Elizabeth’s own feelings from positive
    aversion to actual love. I do not know whether the all-grasping hand of
    the playwright has ever been laid upon_ Pride and Prejudice; _and I dare
    say that, if it were, the situations would prove not startling or
    garish enough for the footlights, the character-scheme too subtle and
    delicate for pit and gallery. But if the attempt were made, it would
    certainly not be hampered by any of those loosenesses of construction,
    which, sometimes disguised by the conveniences of which the novelist can
    avail himself, appear at once on the stage._
    _I think, however, though the thought will doubtless seem heretical to
    more than one school of critics, that construction is not the highest
    merit, the choicest gift, of the novelist. It sets off his other gifts
    and graces most advantageously to the critical eye; and the want of it
    will sometimes mar those graces--appreciably, though not quite
    consciously--to eyes by no means ultra-critical. But a very badly-built
    novel which excelled in pathetic or humorous character, or which
    displayed consummate command of dialogue--perhaps the rarest of all
    faculties--would be an infinitely better thing than a faultless plot
    acted and told by puppets with pebbles in their mouths. And despite the
    ability which Miss Austen has shown in working out the story, I for one
    should put_ Pride and Prejudice _far lower if it did not contain what
    seem to me the very masterpieces of Miss Austen’s humour and of her
    faculty of character-creation--masterpieces who may indeed admit John
    Thorpe, the Eltons, Mrs. Norris, and one or two others to their company,
    but who, in one instance certainly, and perhaps in others, are still
    superior to them._
    _The characteristics of Miss Austen’s humour are so subtle and delicate
    that they are, perhaps, at all times easier to apprehend than to
    express, and at any particular time likely to be differently
    apprehended by different persons. To me this humour seems to possess a
    greater affinity, on the whole, to that of Addison than to any other of
    the numerous species of this great British genus. The differences of
    scheme, of time, of subject, of literary convention, are, of course,
    obvious enough; the difference of sex does not, perhaps, count for much,
    for there was a distinctly feminine element in “Mr. Spectator,” and in
    Jane Austen’s genius there was, though nothing mannish, much that was
    masculine. But the likeness of quality consists in a great number of
    common subdivisions of quality--demureness, extreme minuteness of touch,
    avoidance of loud tones and glaring effects. Also there is in both a
    certain not inhuman or unamiable cruelty. It is the custom with those
    who judge grossly to contrast the good nature of Addison with the
    savagery of Swift, the mildness of Miss Austen with the boisterousness
    of Fielding and Smollett, even with the ferocious practical jokes that
    her immediate predecessor, Miss Burney, allowed without very much
    protest. Yet, both in Mr. Addison and in Miss Austen there is, though a
    restrained and well-mannered, an insatiable and ruthless delight in
    roasting and cutting up a fool. A man in the early eighteenth century,
    of course, could push this taste further than a lady in the early
    nineteenth; and no doubt Miss Austen’s principles, as well as her heart,
    would have shrunk from such things as the letter from the unfortunate
    husband in the_ Spectator, _who describes, with all the gusto and all the
    innocence in the world, how his wife and his friend induce him to play
    at blind-man’s-buff. But another_ Spectator _letter--that of the damsel
    of fourteen who wishes to marry Mr. Shapely, and assures her selected
    Mentor that “he admires your_ Spectators _mightily”--might have been
    written by a rather more ladylike and intelligent Lydia Bennet in the
    days of Lydia’s great-grandmother; while, on the other hand, some (I
    think unreasonably) have found “cynicism” in touches of Miss Austen’s
    own, such as her satire of Mrs. Musgrove’s self-deceiving regrets over
    her son. But this word “cynical” is one of the most misused in the
    English language, especially when, by a glaring and gratuitous
    falsification of its original sense, it is applied, not to rough and
    snarling invective, but to gentle and oblique satire. If cynicism means
    the perception of “the other side,” the sense of “the accepted hells
    beneath,” the consciousness that motives are nearly always mixed, and
    that to seem is not identical with to be--if this be cynicism, then
    every man and woman who is not a fool, who does not care to live in a
    fool’s paradise, who has knowledge of nature and the world and life, is
    a cynic. And in that sense Miss Austen certainly was one. She may even
    have been one in the further sense that, like her own Mr. Bennet, she
    took an epicurean delight in dissecting, in displaying, in setting at
    work her fools and her mean persons. I think she did take this delight,
    and I do not think at all the worse of her for it as a woman, while she
    was immensely the better for it as an artist.
    ```

```python
from instructor import Instructor, Mode, patch
from anthropic import Anthropic
from pydantic import BaseModel

client = Instructor( # (1)!
    client=Anthropic(),
    create=patch(
        create=Anthropic().beta.prompt_caching.messages.create,
        mode=Mode.ANTHROPIC_TOOLS,
    ),
    mode=Mode.ANTHROPIC_TOOLS,
)


class Character(BaseModel):
    name: str
    description: str


with open("./book.txt", "r") as f:
    book = f.read()

resp = client.chat.completions.create(
    model="claude-3-haiku-20240307",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<book>" + book + "</book>",
                    "cache_control": {"type": "ephemeral"}, # (2)!
                },
                {
                    "type": "text",
                    "text": "Extract a character from the text given above",
                },
            ],
        },
    ],
    response_model=Character,
    max_tokens=1000,
)
```

1. Since the feature is still in beta, we need to manually pass in the function that we're looking to patch.

2. Anthropic requires that you explicitly pass in the `cache_control` parameter to indicate that you want to cache the content.

!!! Warning "Caching Considerations"

    **Minimum cache size**: For Claude Haiku, your cached content needs to be a minimum of 2048 tokens. For Claude Sonnet, the minimum is 1024 tokens.

**Benefits**: The cost of reading from the cache is 10x lower than if we were to process the same message again and enables us to execute our queries significantly faster.

We've written a more detailed blog on how to use the `create_with_completion` method [here](../blog/posts/anthropic-prompt-caching.md) to validate you're getting a cache hit with instructor.
