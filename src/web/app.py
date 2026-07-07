import os
import sys
import json
import requests
from flask import Flask, render_template, request, jsonify

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

# Initialize Flask app
app = Flask(__name__)

# API URL
API_URL = f"http://127.0.0.1:{config.API_PORT}"


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request
    """
    try:
        # Get form data
        transaction_data = {
            'trans_date_trans_time': request.form.get('trans_date_trans_time'),
            'cc_num': request.form.get('cc_num'),
            'merchant': request.form.get('merchant'),
            'category': request.form.get('category'),
            'amt': float(request.form.get('amt')),
            'first': request.form.get('first'),
            'last': request.form.get('last'),
            'gender': request.form.get('gender'),
            'street': request.form.get('street'),
            'city': request.form.get('city'),
            'state': request.form.get('state'),
            'zip': request.form.get('zip'),
            'lat': float(request.form.get('lat')),
            'long': float(request.form.get('long')),
            'city_pop': int(request.form.get('city_pop')),
            'job': request.form.get('job'),
            'dob': request.form.get('dob'),
            'trans_num': request.form.get('trans_num'),
            'unix_time': int(request.form.get('unix_time')),
            'merch_lat': float(request.form.get('merch_lat')),
            'merch_long': float(request.form.get('merch_long'))
        }

        # Call API
        response = requests.post(f"{API_URL}/predict", json=transaction_data)

        if response.status_code == 200:
            result = response.json()
            return render_template('result.html', result=result, transaction=transaction_data)
        else:
            error_message = f"API Error: {response.status_code} - {response.text}"
            return render_template('error.html', error=error_message)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api-status')
def api_status():
    """
    Check API status
    """
    try:
        response = requests.get(f"{API_URL}/health")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/model-info')
def model_info():
    """
    Get model information
    """
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            return render_template('model_info.html', model_info=response.json())
        else:
            error_message = f"API Error: {response.status_code} - {response.text}"
            return render_template('error.html', error=error_message)
    except Exception as e:
        return render_template('error.html', error=str(e))


def main():
    """
    Run the web server
    """
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=True
    )


if __name__ == "__main__":
    main()
