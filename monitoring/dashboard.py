"""
Monitoring dashboard for the trading system.
This module creates a simple web server (using Flask) to display key system metrics.
"""

from flask import Flask, jsonify, render_template
from core.config import config
from data.database import db
from core.logger import log_manager
import datetime

app = Flask(__name__)
logger = log_manager.get_logger(__name__)

@app.route('/')
def dashboard():
    # For demonstration, we return some dummy metrics.
    metrics = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "open_positions": 5,  # In production, query your execution/position module or database
        "risk_level": "normal",
        "equity": 10000,
        "daily_pnl": 200
    }
    return render_template('dashboard.html', metrics=metrics)

@app.route('/api/metrics')
def metrics_api():
    metrics = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "open_positions": 5,
        "risk_level": "normal",
        "equity": 10000,
        "daily_pnl": 200
    }
    return jsonify(metrics)

if __name__ == '__main__':
    logger.info("Starting monitoring dashboard on port 5000")
    app.run(host='0.0.0.0', port=5000)
