from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from model import get_mentor_recommendation

# Setup connection
load_dotenv(os.path.join(os.getcwd(), '.env'))

try:
    connection_string = "mysql+mysqlconnector://{0}:{1}@{2}:3306/{3}".format(
        os.getenv('DB_USERNAME'),
        os.getenv('DB_PASSWORD'),
        os.getenv('DB_HOST'),
        os.getenv('DB_DATABASE')
    )
    engine = create_engine(connection_string)
    connection = engine.connect()
except:
    raise Exception("Gabisa konek DB")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "API is Online"

@app.route("/get-mentors", methods=['POST'])
def get_mentors():
    user_id = request.json['user_id']
    user_text = request.json['user_text']
    mentors = get_mentor_recommendation(connection, user_id, user_text)
    return jsonify(mentors)