from flask import render_template, Blueprint, request, current_app, send_file, send_from_directory
from flask_sock import Sock


# import command to stringify a json object
import json
import os

html = Blueprint('html', __name__, template_folder='templates')


@html.route('/')
def index():
    return render_template('index.html')
