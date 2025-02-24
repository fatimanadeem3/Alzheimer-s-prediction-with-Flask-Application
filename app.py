import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib 
app = Flask(__name__)


UPLOAD_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        name = request.form['name']
        gender = 1 if request.form['gender'] == 'male' else 0
        age = float(request.form['age'])
        education_level = {'primary': 1, 'secondary': 2, 'higher': 3}[request.form['educationLevel']]
        physical_activity = 1 if request.form['physicalActivity'] == 'yes' else 0
        smoking_status = 1 if request.form['smokingStatus'] == 'yes' else 0
        alcohol_consumption = 1 if request.form['alcoholConsumption'] == 'yes' else 0
        diabetes = 1 if request.form['diabetes'] == 'yes' else 0
        hypertension = 1 if request.form['hypertension'] == 'yes' else 0
        cholesterol = float(request.form['cholesterolLevel'])
        family_history = 1 if request.form['familyHistory'] == 'yes' else 0
        cognitive_score = float(request.form['cognitiveScore'])
        depression_level = 1 if request.form['depressionLevel'] == 'yes' else 0
        sleep_quality = 1 if request.form['sleepQuality'] == 'yes' else 0
        dietary_habits = 1 if request.form['dietaryHabits'] == 'yes' else 0
        air_pollution = 1 if request.form['airPollution'] == 'yes' else 0
        genetic_risk = 1 if request.form['geneticRisk'] == 'yes' else 0
        social_engagement = 1 if request.form['socialEngagement'] == 'yes' else 0
        stress_levels = 1 if request.form['stressLevels'] == 'yes' else 0

        
        input_features = np.array([[
            age, gender, education_level, physical_activity, smoking_status,
            alcohol_consumption, diabetes, hypertension, cholesterol,
            family_history, cognitive_score, depression_level, sleep_quality,
            dietary_habits, air_pollution, genetic_risk, social_engagement,
            stress_levels
        ]])

        
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[0][1] * 100

        
        if prediction[0] == 1:
            diagnosis = "Probability of Alzheimer's"
            recommendation = "We recommend consulting a neurologist for further assessment and cognitive therapy options. Early intervention may improve quality of life.Thank You!"
        else:
            diagnosis = "No Alzheimer's detected"
            recommendation = "Congratulations! Your cognitive health appears to be in good condition. Maintain a healthy lifestyle and regular check-ups.Thank You!"

        result_text = f"""
        Alzheimer's Prediction Report
        --------------------------------------
        Name: {name}
        Age: {age}
        Gender: {'Male' if gender == 1 else 'Female'}
        Education Level: {education_level}
        Physical Activity: {'Yes' if physical_activity == 1 else 'No'}
        Smoking Status: {'Yes' if smoking_status == 1 else 'No'}
        Alcohol Consumption: {'Yes' if alcohol_consumption == 1 else 'No'}
        Diabetes: {'Yes' if diabetes == 1 else 'No'}
        Hypertension: {'Yes' if hypertension == 1 else 'No'}
        Cholesterol Level: {cholesterol}
        Family History: {'Yes' if family_history == 1 else 'No'}
        Cognitive Score: {cognitive_score}
        Depression Level: {'Yes' if depression_level == 1 else 'No'}
        Sleep Quality: {'Yes' if sleep_quality == 1 else 'No'}
        Dietary Habits: {'Yes' if dietary_habits == 1 else 'No'}
        Air Pollution Exposure: {'Yes' if air_pollution == 1 else 'No'}
        Genetic Risk Factor: {'Yes' if genetic_risk == 1 else 'No'}
        Social Engagement: {'Yes' if social_engagement == 1 else 'No'}
        Stress Levels: {'Yes' if stress_levels == 1 else 'No'}
        --------------------------------------
        Prediction: {diagnosis} with a probability of {probability:.2f}%

        Doctor's Recommendation:
        {recommendation}
        """

        report_filename = f"{name}_alzheimers_report.txt"
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
        with open(report_path, 'w') as file:
            file.write(result_text)

        return jsonify({
            'prediction': diagnosis,
            'probability': probability,
            'download_url': f'/download/{report_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download/<filename>')
def download_report(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
