# python_ai_test

Simple Python 3 ML test. APIs and Frameworks used: 
- Flask
- Boto3
- Pandas
- Numpy
- Keras
- TensorFlow
- SKLearn
- PyMongo InMemory
- Python-dotEnv
- Joblib

---

# Index

* [ML Model](#ML-Model)
* [MongoDB](#MongoDB)
* [Flask API](#Flask-API)
* [Configure](#Configure)
* [Execute](#Execute)

---

## ML Model

The data (csv format) is automatically downloaded from an AWS S3 bucket, using the configuration of the .env file (which is not uploaded for privacy reasons). Please, see the [Execute](#execute) section for more information.

Once downloaded, the file is pre-processed and splited (train and test datasets).

I have choosen to train a simple _Sequential_ deep learning model for to predict a new value. For this solution I choosed the Long Short-Term Momory network because it is very powerful in sequence problems since they are able to store past information.

> Note: The training have been made just as demo purpouse. It is not well trained, nor fited. The intent in this exercise is to show the technique and mechanisms for that kind of tasks.

Al the data is normalized using the _MinMaxScaler_, with values between 0 and 1.

The layers used to train the model are:
- LSTM (50 units)
- Dropout (20%)
- LSTM (50 units)
- Dropout (20%)
- LSTM (50 units)
- Dropout (20%)
- LSTM (50 units)
- Dropout (20%)
- Dense (1 unit)

The loss function used is the _mean squared error_, and the optimizer is _ADAM_.

Once the training process is finished both, the model and the scaler, are saved on disk for future use.

[Return](#index)

---

## MongoDB

Just for the test to be portable, I used a _in memory_ version of MongoDB.

[Return](#index)

---

## Flask API

The REST API presents three resources to be consumed:

- **GET** /api/v1/rows: returns a _json_ document with the number of rows in the sample csv file.
```json
{ "row": 22390 }
```
- **GET** /api/v1/tip/average: returns a _json_ document with the average value for the `tip_amout` column.
```json
{ "tipAverage": 2.4 }
```
- **POST** /api/v1/predict: returns a _json_ document with the predictions based on the values.
```json
{ "predictions": [[3.898], [1.234], [5.56]] }
```

[Return](#index)

---

## Configure

It is recommended to create a python `venv` for to execute the app, so that the requirement could not affect to any other application:

```bash
python3 -m venv ai_test
```

Once created, activate it and install the project dependencies:

```bash
source ai_test/bin/activate
pip3 install -r requirements.txt
```

[Return](#index)

---

## Execute

> Before running the app, it is mandatory to create a `.env` file. See [run.sh](./run.sh) for more details.

Execute the script `run.sh` in order to start the application:

```bash
./run.sh
```

[Return](#index)

---
