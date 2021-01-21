from flask import Flask, Blueprint, jsonify, Response, request
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
        response = jsonify({ 'rows': row_count})
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

@bp.route('/tip/average', methods = ['GET'])
@cross_origin()
def tip_avg():
    row_count, tip_avg = app.model.get_stats()
    return Response(
        response = jsonify({ 'tipAverage': tip_avg})
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

@bp.route('/predict', methods = ['GET'])
@cross_origin()
def predict():
    data = request.json
    prediction = app.model.predict(data.values)
    return Response(
        response = jsonify({ 'tipAverage': tip_avg})
        , status = 200
        , headers = {'Access-Control-Allow-Origin': '*'}
    )

app.register_blueprint(bp, url_prefix = '/api/v1')
