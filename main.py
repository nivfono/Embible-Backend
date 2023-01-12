from flask import Flask, request,jsonify
from flask_cors import CORS
import config
from src.ensamble import Ensamble
from huggingface_hub import login


app = Flask(__name__)
CORS(app)
login(config.configs['hf_token'])
ens=Ensamble()
@app.route("/calc",methods=['GET'])
def calc():
    args=request.args.to_dict()
    return jsonify(ens.predict(args['text']))

# app.run(host="0.0.0.0", port=443,debug=True,ssl_context='adhoc')
app.run(host="0.0.0.0", port=8000,debug=True)