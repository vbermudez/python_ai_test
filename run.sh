export FLASK_APP=main.py
# to download CSV and train a model, the app requires this environment variables, in a .env file:
# export S3_REGION=
# export S3_BUCKET=
# export S3_FILE=
# export S3_KEY=
# export S3_SECRET=

flask run