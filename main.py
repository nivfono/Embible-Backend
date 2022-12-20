
from flask import Flask, request,jsonify

from flask_cors import CORS

#
from src.model import Model

app = Flask(__name__)
CORS(app)
# @app.route("/",methods=['GET'])
# def hello_world():
#     return render_template('index.html')

@app.route("/mock",methods=['GET'])
def mock():
    m=Model()
    return jsonify(m.getMock())

app.run(host="localhost", port=8000,debug=True,)
