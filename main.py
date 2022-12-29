
from flask import Flask, request,jsonify

from flask_cors import CORS

#
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
    # m=Model(config.configs['char_model_path'])
    # return jsonify(m.calc(args['text']))
    return jsonify(ens.predict(args['text']))

app.run(host="localhost", port=443,debug=True,)
