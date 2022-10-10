from fastapi import FastAPI
from routes.tmr_api import prediction
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

origins = ["http://127.0.0.1:5500","*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# model route
app.include_router(prediction)


