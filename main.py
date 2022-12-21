
from flask import Flask, request,jsonify

from flask_cors import CORS

#
from src.model import Model
from config import configs
app = Flask(__name__)
CORS(app)
# @app.route("/",methods=['GET'])
# def hello_world():
#     return render_template('index.html')

@app.route("/calc",methods=['GET'])
def calc():
    m=Model(configs['model_path'])
    args=request.args.to_dict()

    print(args)

    return jsonify(m.calc(args['text']))

app.run(host="localhost", port=8000,debug=True,)
