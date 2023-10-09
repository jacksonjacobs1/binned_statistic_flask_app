from flask import Flask
# from flask_restless import APIManager
import argparse

from scatterHTML import html
from scatterAPI import scatterAPI

app = Flask(__name__)
app.register_blueprint(html)
app.register_blueprint(scatterAPI)
app.logger_name = 'flask'

if __name__ == '__main__':
    
    app.logger.info('Starting Flask app')
    app.run(host='0.0.0.0', port=5555, debug=False)