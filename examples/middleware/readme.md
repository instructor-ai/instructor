# Middleware in Instructor

Middleware in Instructor allows you to modify the messages sent to the language model before they are processed. This is beneficial because it enables you to perform custom preprocessing, add context, or even implement simple retrieval-augmented generation (RAG) techniques.

Middleware can be defined as simple functions or classes (when you need stateful variables). They are then registered with the Instructor client using the `with_middleware` method.

## what is middleware?

Middleware is a way to modify the input or output of a function or method. In the context of language models and AI assistants, middleware allows you to intercept and modify the messages being sent to the model before they are processed.

Some common use cases for middleware include:

- Preprocessing the input messages (e.g. cleaning up text, adding context)
- Implementing retrieval augmented generation by fetching relevant information and appending it to the messages
- Filtering or moderating content 
- Logging or monitoring the messages being sent to the model

Middleware functions take in the list of messages, make any desired changes, and return the modified list of messages to be sent to the model.

Instructor makes it easy to define and use middleware. You can create middleware as simple functions using the `@messages_middleware` decorator, or for more complex stateful middleware you can define a class that inherits from `MessageMiddleware` and implements the `__call__` method.

Once defined, middleware is registered with the Instructor client using the `with_middleware()` method. This allows chaining multiple middleware together.

## Simple RAG Example

Middleware can also be used to implement more advanced techniques like retrieval-augmented generation (RAG). RAG involves retrieving relevant information from an external source and using it to augment the input to the language model. This can help provide additional context and improve the quality and accuracy of the generated responses.

To implement a simple RAG middleware, you could define a function or class that takes the input messages, performs a retrieval step to find relevant information, and then appends that information to the messages before sending them to the model. For example:

```python
@instructor.messages_middleware
def add_retrieval_augmentation(messages):
    # Perform retrieval step to find relevant information
    relevant_information = retrieve_relevant_information(messages)

    # Append the relevant information to the messages
    return messages + [{
        "role": "user",
        "content": f"Relevant Information: {relevant_information}"
    }]
```

## Logging and Monitoring
Another useful application of middleware is for logging and monitoring the messages being sent to and received from the language model. This can be helpful for debugging, auditing, or analyzing the conversations.

To implement logging middleware, you can define a function or class that takes the input messages, logs them to a file or database, and then returns the original messages unmodified. For example:

```python
@instructor.messages_middleware
def logging_middleware(messages):
    import logging
    logging.info(f"Input messages: {messages}")
    
    # Return the original messages unmodified
    return messages
```

## Stateful Middleware

For more advanced stateful middleware, you can define a class that inherits from `MessageMiddleware` and implements the `__call__` method. This allows you to maintain state across multiple calls to the middleware.

For example, let's say you want to implement a middleware that adds user preferences to the messages. You could define a stateful middleware class like this:

```python
class UserPreferencesMiddleware(MessageMiddleware):

    user_id: str

    def __call__(self, messages):
        preferences = get_user_preferences(self.user_id)
        for message in messages:
            if message.role == "system":
                message.content += f"\n\nUser Preferences: {preferences}"
        return messages
```

As you can see above, middleware provides a flexible way to modify and augment the messages being sent to and received from the language model. This can be used for a variety of purposes, such as adding relevant information, logging and monitoring conversations, and maintaining stateful interactions.