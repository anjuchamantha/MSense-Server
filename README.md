# MSense-Server

## Install the dependencies

pip install -r requirements.txt

## Run the live API server
In terminal (main.py level): \
uvicorn main:app --reload

## Heroku Deploy
heroku login
heroku git:remote -a msense-server
heroku git:remote -a msense-server
