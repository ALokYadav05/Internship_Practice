from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hellow_world():
    return render_template('index.html',message='Hello, World!')

@app.route('/about')
def about():
    return render_template('about.html', msg='This is about-section')

@app.route('/user/<name>')  #dyanamuic-routing
def user(name):
    return render_template('user.html', m = f"hello {name}!")


if __name__ == "__main__":
    app.run(debug=True)

