# Fastapi MongoDB REST API

## Create a new directory on your local machine
```
mkdir <directory-name>
```
## Go inside the directory
```
cd  <directory-name>
```
## Clone the remote repository
```
git clone <repository-link>
```
## Setup env
```
virtualenv venv
```
## For Windows
```
source venv/Scripts/activate
```
## For Linux/Mac
```
source venv/bin/activate
```
<p align="center">
  <img width="700" height="100" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/1_fastapi.png">
</p>

## Checkout to TMR_fastAPI branch
```
git checkout TMR_fastAPI
```
## Install packages using requirement.txt file
```
pip install -r requirement.txt
```
## Run this command in the current directory ie. FastAPI
```
 . fast_env.sh
```
## Start server 
```
uvicorn index:app --reload
```
<p align="center">
  <img width="700" height="100" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/2_fastapi.png">
</p>

## Open the browser and type the following url in the search bar.
```
http://127.0.0.1:8000/docs#/default/
```
### Snapshots of each APIs
<p align="center">
  <img width="700" height="700" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/3_fastapi.png">
</p>

## API to display welcome message
<p align="center">
  <img width="700" height="700" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/4_fastapi.png">
</p>

## API to predict the TechCategory based on the 9 input features
<p align="center">
  <img width="700" height="700" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/5_fastapi.png">
</p>

<p align="center">
  <img width="700" height="100" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/6_fastapi.png">
</p>

## API to predict the TechCategory based on the input features taken from a csv file.
## The csv file must contain these 9 features to make predictions['RFP Last Internal Rating', 'Trial Last Internal Rating', 'Trial % Present', 'RFP Avg TRACK Score', 'Trial Avg TRACK Score', 'RFP Tech Ability', 'RFP Last TRACK Score', 'Trial Techability Score', 'Practice Head']

<p align="center">
  <img width="700" height="700" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/8_fastapi.png">
</p>

<p align="center">
  <img width="700" height="700" src="https://github.com/Neelesh-BL/TMR/blob/TMR_fastAPI/FastAPI/images/7_fastapi.png">
</p>