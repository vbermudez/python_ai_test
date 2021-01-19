import sys
import logging

#from s3 import read_config, download_file
from ml import Model
from api import app

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
root.addHandler(handler)

model = Model('/home/vbermudez/test/python_ai_test/csv/data.csv', '/home/vbermudez/test/python_ai_test/model/lstm_model.h5')

if not model.exists: model.train()

row_count, tip_avg = model.get_stats()

print(f'LINEAS={row_count}')
print(f'AVERAGE tip_amout={tip_avg}')

model._test_model()
