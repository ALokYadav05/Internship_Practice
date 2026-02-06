from flask import Flask, jsonify
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/hello', methods=['GET'])
def hello():
    """
    A simple hello world endpoint.
    ---
    responses:
     200:
      description: Returns a greeting message
      examples:
        application/json: {"message": "Hello, World!"}
    """
    return jsonify(message="Hello, World!")


if __name__ == "__main__":
    app.run(debug=True)