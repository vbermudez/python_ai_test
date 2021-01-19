export FLASK_APP=main.py
# to train a model, the app requires this environment variables:
# export S3_REGION=
# export S3_BUCKET=
# export S3_FILE=
# export S3_KEY=
# export S3_SECRET=
[ -f .env ] && source .env

#flask run
python3 main.py