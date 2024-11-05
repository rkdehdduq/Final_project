from pymongo import MongoClient
from openai import OpenAI
import os
from dotenv import load_dotenv
import certifi

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
#  MongoDB Atlas 연결
uriCloud=os.getenv("MONGODB_URI")
cert = certifi.where()


# 사용할 데이터베이스와 컬렉션 선택
dbclient = MongoClient(uriCloud, tls=True, tlsCAFile=cert, connectTimeoutMS=30000, socketTimeoutMS=30000)

