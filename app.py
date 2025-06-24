import os
from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
import pdfkit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app with templates folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
logger.info(f"Template directory: {template_dir}")
if not os.path.exists(template_dir):
    logger.error(f"Templates folder does not exist at: {template_dir}")
app = Flask(__name__, template_folder=template_dir)

# 5MB file limit
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv(df):
    if df.columns.empty or df.columns.has_duplicates:
        return False, "CSV must have valid, unique headers."
    if not df.select_dtypes(include=np.number).columns.tolist():
        return False, "CSV must contain at least one numeric column."
    return True, None

def run_fraud_detection(df_input, contamination_factor=0.02):
    df = df_input.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_column_names = ['is_fraud', 'fraud_flag', 'fraud_indicator']
    actual_target_column = None
    for tc in target_column_names:
        if tc in numeric_cols:
            numeric_cols.remove(tc)
            actual_target_column = tc
            break

    if not numeric_cols:
        return {"error": "No numeric columns found for anomaly detection."}, None

    X = df[numeric_cols].copy()

    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info(f"Filled missing values in '{col}' with median: {median_val}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination_factor, random_state=42)
    iso_forest.fit(X_scaled_df)
    df['iso_forest_anomaly_score'] = iso_forest.decision_function(X_scaled_df)
    df['iso_forest_is_fraud'] = np.where(iso_forest.predict(X_scaled_df) == -1, 1, 0)

    # One-Class SVM
    oc_svm = OneClassSVM(nu=contamination_factor, kernel="rbf", gamma='auto')
    oc_svm.fit(X_scaled_df)
    df['oc_svm_anomaly_score'] = oc_svm.decision_function(X_scaled_df)
    df['oc_svm_is_fraud'] = np.where(oc_svm.predict(X_scaled_df) == -1, 1, 0)

    # Final prediction
    df['final_is_fraud_prediction'] = (df['iso_forest_is_fraud'] + df['oc_svm_is_fraud']).apply(
        lambda x: 1 if x >= 1 else 0
    )

    results = df.to_dict(orient='records')
    summary_data = {
        'total_transactions': len(df),
        'predicted_fraud_count': int(df['final_is_fraud_prediction'].sum())
    }

    classification_report_str = ""
    confusion_matrix_image_base64 = None

    if actual_target_column:
        y_true = df[actual_target_column]
        y_pred = df['final_is_fraud_prediction']
        classification_report_str = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Confusion matrix image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
                    yticklabels=['Actual Not Fraud', 'Actual Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        confusion_matrix_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    return {
        "results": results,
        "summary": summary_data,
        "report": classification_report_str,
        "confusion_matrix_image": confusion_matrix_image_base64
    }, actual_target_column

@app.route('/', methods=['GET'])
def upload_form():
    template_path = os.path.join(template_dir, 'index.html')
    logger.info(f"Attempting to render template: {template_path}")
    if not os.path.exists(template_path):
        logger.error(f"Template file not found: {template_path}")
        return jsonify({"error": "Template file not found."}), 500
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {str(e)}")
        return jsonify({"error": "Failed to load the upload form."}), 500
    

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        # Extract data from the POST request
        data = request.json

        # Render the HTML using your Jinja2 template
        rendered_html = render_template(
    'report_template.html',
    filename=data.get('filename', 'Uploaded File'),
    summary=data.get('summary', {}),
    classification_report=data.get('classification_report', ''),
    confusion_matrix_image=data.get('confusion_matrix_image', ''),
    generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    results=data.get('results', [])  # ADD THIS LINE
)

        # Hardcoded path to wkhtmltopdf.exe
        config = pdfkit.configuration(
            wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        )

        # Generate PDF
        pdf = pdfkit.from_string(rendered_html, False, configuration=config)

        # Return the PDF as a downloadable file
        return send_file(
            io.BytesIO(pdf),
            mimetype='application/pdf',
            download_name='fraud_analysis_report.pdf',
            as_attachment=True
        )

    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}")
        return {"error": f"Failed to generate PDF: {str(e)}"}, 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are allowed."}), 400

    try:
        df = pd.read_csv(file)
        logger.info(f"File '{file.filename}' loaded successfully.")

        is_valid, error_message = validate_csv(df)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        if len(df) > 100000:
            df = df.head(100000)
            logger.info("Limited to first 100,000 rows for performance.")

        detection_output, target_col_used = run_fraud_detection(df)

        if "error" in detection_output:
            return jsonify(detection_output), 400

        response_data = {
            "message": "File processed successfully!",
            "filename": file.filename,
            "summary": detection_output['summary'],
            "results": detection_output['results']
        }

        if detection_output['report']:
            response_data['classification_report'] = detection_output['report']
        if detection_output['confusion_matrix_image']:
            response_data['confusion_matrix_image'] = detection_output['confusion_matrix_image']
        response_data['target_column_identified'] = target_col_used

        return jsonify(response_data), 200

    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV file is empty."}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Invalid CSV format. Ensure it is comma-separated with valid headers."}), 400
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
