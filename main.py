
from flask import Flask, request,jsonify

from flask_cors import CORS

#
from src.ensamble import Ensamble
from src.model.model import Model
from config import configs
app = Flask(__name__)
CORS(app)
# @app.route("/",methods=['GET'])
# def hello_world():
#     return render_template('index.html')
ens=Ensamble()
@app.route("/calc",methods=['GET'])
def calc():
    args=request.args.to_dict()
    return jsonify(ens.predict(args['text']))

app.run(host="localhost", port=8000,debug=True,)
