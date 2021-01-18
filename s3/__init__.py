import os
import logging

from boto3.session import Session
import boto3

def read_config():
    return {
        'region': os.getenv('S3_REGION')
        , 'key': os.getenv('S3_KEY')
        , 'secret': os.getenv('S3_SECRET')
        , 'bucket': os.getenv('S3_BUCKET')
        , 'filename': os.getenv('S3_FILE')
        , 'local_file': os.path.join(os.getcwd(), 'csv', 'data.csv')
    }

def download_file():
    cfg = read_config()

    logging.info(f'Downloading {cfg["filename"]} fomr S3 Bucket {cfg["bucket"]}')

    s3 = boto3.resource('s3'
        , cfg['region']
        , aws_access_key_id = cfg['key'] 
        , aws_secret_access_key = cfg['secret']
    )
    s3.Object(cfg['bucket'], cfg['filename']).download_file(cfg['local_file'])
