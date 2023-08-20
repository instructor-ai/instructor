import openai
from sqlalchemy import create_engine
from patch_sql import instrument_with_sqlalchemy

engine = create_engine("sqlite:///chat.db", echo=True)
instrument_with_sqlalchemy(engine)

resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {
            "role": "user",
            "content": "1+1",
        }
    ],
)
"""
2023-08-19 14:57:04,339 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2023-08-19 14:57:04,340 INFO sqlalchemy.engine.Engine INSERT INTO chatcompletion 
2023-08-19 14:57:04,340 INFO sqlalchemy.engine.Engine [generated in 0.00021s] ('chatcmpl-7pOEKM94WoZX9IJ9rRzPacrylzlOR', '2023-08-19 21:57:04.340462', None, ...
2023-08-19 14:57:04,341 INFO sqlalchemy.engine.Engine INSERT INTO message
2023-08-19 14:57:04,342 INFO sqlalchemy.engine.Engine [generated in 0.00012s]
2023-08-19 14:57:04,342 INFO sqlalchemy.engine.Engine INSERT INTO message 
2023-08-19 14:57:04,342 INFO sqlalchemy.engine.Engine [cached since 0.0004963s ago]
2023-08-19 14:57:04,342 INFO sqlalchemy.engine.Engine COMMIT
"""