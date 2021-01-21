import sys
import logging

from ml import Model

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
root.addHandler(handler)

model = Model('/home/vbermudez/test/python_ai_test/csv/data.csv', '/home/vbermudez/test/python_ai_test/model/lstm_model.h5')

if not model.exists: model.train()
else: model.prepare_model()

row_count, tip_avg = model.get_stats()

logging.info(f'LINEAS={row_count}')
logging.info(f'AVERAGE tip_amout={tip_avg}')

values = [[1.70, 8]]

logging.info( f'Predictions:\n{model.predict( values )}' )

from api import app

app.model = model
