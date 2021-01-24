import logging
from datetime import datetime
from flask import Flask, Blueprint, Response, request, json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
bp = Blueprint('v1_api', __name__)

app.model = None
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return 'Simple ML API'

@bp.route('/rows', methods = ['GET'])
@cross_origin()
def rows():
    row_count, tip_avg = app.model.get_stats()
    return Response(
        response = json.dumps({ 'rows': row_count })
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

@bp.route('/tip/average', methods = ['GET'])
@cross_origin()
def tip_avg():
    row_count, tip_avg = app.model.get_stats()
    return Response(
        response = json.dumps({ 'tipAverage': tip_avg })
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

@bp.route('/predict', methods = ['POST'])
@cross_origin()
def predict():
    data = json. request.get_json()
    logging.info(f'RECV JSON:\n{data}')
    predictions = app.model.predict( data['values'] )
    app.db.predictions.insert_one({
        'values': data['values']
        , 'predictions': predictions.tolist()
        , 'date': datetime.now()
    })
    return Response(
        response = json.dumps({ 'predictions': predictions.tolist() })
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

app.register_blueprint(bp, url_prefix = '/api/v1')
