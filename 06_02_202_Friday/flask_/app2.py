from flask import Flask, render_template, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)





if __name__ == '__main__':
    app.run(debug=True)

