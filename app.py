from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('maternal_health_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Will load from /templates folder

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     features = [
#         data['age'],
#         data['systolic'],
#         data['diastolic'],
#         data['blood_sugar'],
#         data['body_temp'],
#         data['heart_rate']
#     ]

#     prediction = model.predict([features])[0]
#     # return jsonify({'prediction': prediction})
#     # return jsonify({'prediction': str(prediction)})
#     return jsonify({'prediction': int(prediction)})
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         features = [
#             data['age'],
#             data['systolic'],
#             data['diastolic'],
#             data['blood_sugar'],
#             data['body_temp'],
#             data['heart_rate']
#         ]
#         prediction = model.predict([features])[0]

#         # Optional: Convert to label
#         labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
#         return jsonify({'prediction': labels[int(prediction)]})
    
#     except Exception as e:
#         print("Prediction error:", e)
#         return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['age'],
            data['systolic'],
            data['diastolic'],
            data['blood_sugar'],
            data['body_temp'],
            data['heart_rate']
        ]

        prediction = model.predict([features])[0]

        # 🔐 Safely convert to Python int and map to label
        labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        risk_label = labels[int(np.round(prediction).item())]  # .item() ensures native Python type

        return jsonify({'prediction': risk_label})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
