"""
Turtle Trading Strategy Flask API
================================

A comprehensive Flask API for the Turtle Trading Strategy project.
Provides endpoints for model prediction, full analysis, and stakeholder outputs.

Author: Panwei Hu
Date: August 27, 2025
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.model_utils import TurtleModelManager, load_latest_model, create_sample_prediction_data
from src.config import get_key

app = Flask(__name__)

# Global model manager
model_manager = None

def initialize_models():
    """Initialize models on startup."""
    global model_manager
    
    # Try to load existing models
    model_manager = load_latest_model()
    
    if model_manager is None:
        print("⚠️  No existing models found. Run /run_full_analysis first.")
        model_manager = TurtleModelManager()
    else:
        print("✅ Models loaded successfully")

# Initialize on startup
initialize_models()

@app.route('/')
def home():
    """Home page with API documentation."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Turtle Trading Strategy API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #0066cc; font-weight: bold; }
            .url { color: #333; font-family: monospace; }
            .description { color: #666; }
        </style>
    </head>
    <body>
        <h1>🐢 Turtle Trading Strategy API</h1>
        <p>Welcome to the Turtle Trading Strategy API. This service provides model predictions and analysis capabilities.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/predict</div>
            <div class="description">Make predictions using trained models. Send JSON with 'features' array.</div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/predict/sample</div>
            <div class="description">Get sample prediction data for testing.</div>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/run_full_analysis</div>
            <div class="description">Run complete analysis pipeline and train new models.</div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/model_summary</div>
            <div class="description">Get summary of loaded models and performance.</div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/health</div>
            <div class="description">Check API health and status.</div>
        </div>
        
        <h2>Example Usage:</h2>
        <pre>
# Make a prediction
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [100.5, 98.2, 105.1, 95.3, 2.1, 0.025, 0.3, 0.6]}'

# Run full analysis
curl -X POST http://localhost:5000/run_full_analysis \\
  -H "Content-Type: application/json" \\
  -d '{"data_path": "data/processed/turtle_enhanced_features_20250825_140536.parquet"}'
        </pre>
    </body>
    </html>
    """
    return html

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': model_manager is not None and len(model_manager.models) > 0,
        'available_models': list(model_manager.models.keys()) if model_manager else []
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using trained models.
    
    Expected JSON format:
    {
        "features": [feature1, feature2, ...],
        "model_type": "regression" or "classification" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" key in request'}), 400
        
        features = data['features']
        model_type = data.get('model_type', 'regression')
        
        # Validate features
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be a list'}), 400
        
        # Convert to DataFrame
        feature_names = ['sma_20', 'sma_50', 'donchian_high_20', 'donchian_low_20', 
                        'atr_20', 'volatility_20', 'trend_strength', 'price_position']
        
        if len(features) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features, got {len(features)}'}), 400
        
        X = pd.DataFrame([features], columns=feature_names)
        
        # Make predictions
        predictions = {}
        
        if model_type == 'regression':
            # Try different regression models
            for model_name in ['regression_Ridge', 'regression_Linear', 'regression_Random Forest']:
                if model_name in model_manager.models:
                    pred = model_manager.predict(X, model_name)[0]
                    predictions[model_name.replace('regression_', '')] = float(pred)
        
        elif model_type == 'classification':
            # Try classification models
            for model_name in ['classification_binary_Random Forest', 'classification_binary_Logistic']:
                if model_name in model_manager.models:
                    pred = model_manager.predict(X, model_name)[0]
                    proba = model_manager.predict_proba(X, model_name)[0] if hasattr(model_manager.models[model_name], 'predict_proba') else None
                    predictions[model_name.replace('classification_binary_', '')] = {
                        'prediction': int(pred),
                        'probability': proba.tolist() if proba is not None else None
                    }
        
        else:
            return jsonify({'error': 'Invalid model_type. Use "regression" or "classification"'}), 400
        
        return jsonify({
            'predictions': predictions,
            'features_used': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/sample')
def predict_sample():
    """Get sample prediction data for testing."""
    try:
        # Create sample data
        sample_data = create_sample_prediction_data(n_samples=5)
        
        # Make predictions if models are available
        predictions = {}
        if model_manager and len(model_manager.models) > 0:
            for i, row in sample_data.iterrows():
                X = pd.DataFrame([row.values], columns=row.index)
                
                # Try regression prediction
                for model_name in ['regression_Ridge', 'regression_Linear']:
                    if model_name in model_manager.models:
                        pred = model_manager.predict(X, model_name)[0]
                        predictions[f'sample_{i}_{model_name.replace("regression_", "")}'] = float(pred)
        
        return jsonify({
            'sample_data': sample_data.to_dict('records'),
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_full_analysis', methods=['POST'])
def run_full_analysis():
    """
    Run complete analysis pipeline.
    
    Expected JSON format:
    {
        "data_path": "path/to/data.parquet",
        "output_dir": "reports" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'data_path' not in data:
            return jsonify({'error': 'Missing "data_path" key'}), 400
        
        data_path = data['data_path']
        output_dir = data.get('output_dir', 'reports')
        
        # Validate data path
        if not Path(data_path).exists():
            return jsonify({'error': f'Data file not found: {data_path}'}), 404
        
        # Run analysis
        print(f"🚀 Starting full analysis with data: {data_path}")
        results = model_manager.run_full_analysis(data_path, output_dir)
        
        return jsonify({
            'status': 'success',
            'message': 'Full analysis completed successfully',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_summary')
def model_summary():
    """Get summary of loaded models and performance."""
    try:
        if model_manager is None:
            return jsonify({'error': 'No models loaded'}), 404
        
        summary = model_manager.get_model_summary()
        
        return jsonify({
            'model_summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<filename>')
def get_report(filename):
    """Serve generated reports and outputs."""
    try:
        reports_dir = Path('reports')
        file_path = reports_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports')
def list_reports():
    """List available reports and outputs."""
    try:
        reports_dir = Path('reports')
        
        if not reports_dir.exists():
            return jsonify({'reports': [], 'message': 'No reports directory found'})
        
        reports = []
        for file_path in reports_dir.iterdir():
            if file_path.is_file():
                reports.append({
                    'filename': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return jsonify({
            'reports': reports,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Turtle Trading Strategy API...")
    print("📊 Available endpoints:")
    print("  - GET  / : API documentation")
    print("  - GET  /health : Health check")
    print("  - POST /predict : Make predictions")
    print("  - GET  /predict/sample : Sample prediction data")
    print("  - POST /run_full_analysis : Run full analysis")
    print("  - GET  /model_summary : Model summary")
    print("  - GET  /reports : List available reports")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 