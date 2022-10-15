import pymongo
from config.config import Config

host = Config.host
port = Config.port
user = Config.username
password = Config.password
db_name = Config.database_name

uri = "mongodb://{}:{}@{}:{}/{}".format(user, password, host, port, db_name)
conn = pymongo.MongoClient(uri)
db = conn[db_name]