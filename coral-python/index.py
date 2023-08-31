from flask import Flask, request
from flask import jsonify
import subprocess

app = Flask(__name__)

# Just a health check
@app.route("/")
def url_root():
    return "OK"

@app.route("/inference/1", methods=["GET"])
def inference1():
    command = ["bash","classify1.sh"]
    result = subprocess.run(command,stdout=subprocess.PIPE)
    return str(result.stdout.decode('utf-8'))

@app.route("/inference/2", methods=["GET"])
def inference2():
    command = ["bash","classify2.sh"]
    result = subprocess.run(command,stdout=subprocess.PIPE)
    return str(result.stdout.decode('utf-8'))

@app.route("/inference/3", methods=["GET"])
def inference3():
    command = ["bash","classify3.sh"]
    result = subprocess.run(command,stdout=subprocess.PIPE)
    return str(result.stdout.decode('utf-8'))

@app.route("/inference/4", methods=["GET"])
def inference4():
    command = ["bash","classify4.sh"]
    result = subprocess.run(command,stdout=subprocess.PIPE)
    return str(result.stdout.decode('utf-8'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
