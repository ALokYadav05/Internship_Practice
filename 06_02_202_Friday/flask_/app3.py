from flask import Flask, render_template, request, jsonify,redirect,url_for

app = Flask(__name__)

items = {
    1 : {'name':'Laptop','price':50000},
    2 : {'name':'mobile','price':10000},
    3 : {'name':'tv','price':15000},
}

#get
@app.route('/items', methods=['GET'])
def get_items():
    return items

@app.route('/items', methods=['POST'])
def add_item():
     data = request.get_json()
     new_id = int(len(items) + 1)

     items[new_id] = data
     return redirect(url_for("get_items")) , jsonify({'id': new_id})

@app.route('/items', methods=['DELETE'])
def delete_item():
    data = request.get_json()
    items.pop(data['id'])
    return redirect(url_for("get_items"))





if __name__ == '__main__':
    app.run(debug=True)


