from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        return f"Logged in as {username}"
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)