from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN
import os

API_KEY = os.environ["API_KEY"]
API_KEY_NAME = os.environ["API_KEY_NAME"]

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_query: str = Security(api_key_query)):

    if api_key_query == API_KEY:
        return api_key_query
 
    else:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials. Please enter correct credentials")