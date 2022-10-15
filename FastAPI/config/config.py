import os

class Config:

    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    username = os.environ["DB_USER_NAME"]
    password = os.environ["DB_USER_PASS"]
    database_name = os.environ["DB_NAME"]