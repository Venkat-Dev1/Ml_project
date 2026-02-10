from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('Home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=int(request.form.get('math_score')),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        prediction_pipeline=PredictPipeline()
        pred=prediction_pipeline.predict(final_new_data)
        predicted_score = round(pred[0], 2)
        result_status = "Passed ✅" if predicted_score > 40 else "Failed ❌"
        return render_template('Home.html', predicted_score=predicted_score, result_status=result_status)
    
if __name__=="__main__":
    print("\n" + "="*60)
    print("🚀 Starting Flask Application...")
    print("📍 URL: http://127.0.0.1:5000")
    print("📍 URL: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)