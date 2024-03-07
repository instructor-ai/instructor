import instructor
import openai
import logging

from pydantic import BaseModel


# Set logging to DEBUG
logging.basicConfig(level=logging.DEBUG)

client = instructor.patch(openai.OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)  # type: ignore

""" 
DEBUG:httpx:load_ssl_context verify=True cert=None trust_env=True http2=False
DEBUG:httpx:load_verify_locations cafile='/Users/jasonliu/dev/instructor/.venv/lib/python3.11/site-packages/certifi/cacert.pem'
DEBUG:instructor:Patching `client.chat.completions.create` with mode=<Mode.FUNCTIONS: 'function_call'>
DEBUG:instructor:max_retries: 1
DEBUG:openai._base_client:Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'json_data': {'messages': [{'role': 'user', 'content': 'Extract Jason is 25 years old'}], 'model': 'gpt-3.5-turbo', 'function_call': {'name': 'UserDetail'}, 'functions': [{'name': 'UserDetail', 'description': 'Correctly extracted `UserDetail` with all the required parameters with correct types', 'parameters': {'properties': {'name': {'title': 'Name', 'type': 'string'}, 'age': {'title': 'Age', 'type': 'integer'}}, 'required': ['age', 'name'], 'type': 'object'}}]}}
DEBUG:httpcore.connection:connect_tcp.started host='api.openai.com' port=443 local_address=None timeout=5.0 socket_options=None
DEBUG:httpcore.connection:connect_tcp.complete return_value=<httpcore._backends.sync.SyncStream object at 0x105062c90>
DEBUG:httpcore.connection:start_tls.started ssl_context=<ssl.SSLContext object at 0x100748680> server_hostname='api.openai.com' timeout=5.0
DEBUG:httpcore.connection:start_tls.complete return_value=<httpcore._backends.sync.SyncStream object at 0x101caa150>
DEBUG:httpcore.http11:send_request_headers.started request=<Request [b'POST']>
DEBUG:httpcore.http11:send_request_headers.complete
DEBUG:httpcore.http11:send_request_body.started request=<Request [b'POST']>
DEBUG:httpcore.http11:send_request_body.complete
DEBUG:httpcore.http11:receive_response_headers.started request=<Request [b'POST']>
DEBUG:httpcore.http11:receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', [(b'Date', b'Mon, 12 Feb 2024 14:55:45 GMT'), (b'Content-Type', b'application/json'), (b'Transfer-Encoding', b'chunked'), (b'Connection', b'keep-alive'), (b'access-control-allow-origin', b'*'), (b'Cache-Control', b'no-cache, must-revalidate'), (b'openai-model', b'gpt-3.5-turbo-0613'), (b'openai-organization', b'scribe-ai'), (b'openai-processing-ms', b'483'), (b'openai-version', b'2020-10-01'), (b'strict-transport-security', b'max-age=15724800; includeSubDomains'), (b'x-ratelimit-limit-requests', b'10000'), (b'x-ratelimit-limit-tokens', b'2000000'), (b'x-ratelimit-remaining-requests', b'9999'), (b'x-ratelimit-remaining-tokens', b'1999975'), (b'x-ratelimit-reset-requests', b'6ms'), (b'x-ratelimit-reset-tokens', b'0s'), (b'x-request-id', b'req_f0fa476897ae165fc50fa90b7968595b'), (b'CF-Cache-Status', b'DYNAMIC'), (b'Set-Cookie', b'__cf_bm=e2_yCrwo4frh6Oq4ZufCEhNJ4lSGJ2.MMtk45X8lrMM-1707749745-1-AfWk8CyACc7aZo6GpCI82FBfI/wmPEFZLNO/Cr3eavTW3xKVFCS7G9jvwYTFLXjJr0cttYsXeLAnjwipw18R0Vo=; path=/; expires=Mon, 12-Feb-24 15:25:45 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None'), (b'Set-Cookie', b'_cfuvid=PyVVCGSMxTg1p.woYvHVVC9E3n69faOs5FOxaDdjXOM-1707749745711-0-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None'), (b'Server', b'cloudflare'), (b'CF-RAY', b'8545aca30c1fa22f-YYZ'), (b'Content-Encoding', b'gzip'), (b'alt-svc', b'h3=":443"; ma=86400')])
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
DEBUG:httpcore.http11:receive_response_body.started request=<Request [b'POST']>
DEBUG:httpcore.http11:receive_response_body.complete
DEBUG:httpcore.http11:response_closed.started
DEBUG:httpcore.http11:response_closed.complete
DEBUG:openai._base_client:HTTP Request: POST https://api.openai.com/v1/chat/completions "200 OK"
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete
"""
