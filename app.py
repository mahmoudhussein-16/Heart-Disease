from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the data from the POST request
    input_data = np.array(data['input'])

    # Ensure the input data has the correct number of features
    if input_data.shape[1] != scaler.n_features_in_:
        return jsonify({'error': f'Expected {scaler.n_features_in_} features, got {input_data.shape[1]}'})
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(scaled_input)
    
    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)