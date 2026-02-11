from pymongo import MongoClient


def connection_db():
    """
    This function is used to connect to the database
    :return: MongoClient object
    """
    client = MongoClient('mongodb://localhost:27017')
    # print(client.server_info())   # this is to make sure that server is live
    print("Mongodb connection is Successful!!")
    return client


def create_client(client, db_name, coll_name):
    """
    This function is used to create the client
    :param client: mongo-client object
    :param db_name: database name
    :param coll_name: collection  name
    :return: object of collection
    """
    db = client[db_name]
    coll = db[coll_name]

    return coll


def insert_one_data(collect, data):
    """
    This function is used to insert data. One data at a time!
    """
    collect.insert_one(data)
    print("Data inserted successfully!!")


def insert_many_data(collect, data):
    """
    This function is used to insert data. Multiple data at a time!
    """
    dt = collect.insert_many(data)
    print(f"{len(dt.inserted_ids)} data record inserted successfully!!")


def read_data(collect):
    """
    This function is used to read data.
    """
    data = collect.find()   # returns the cursor which is object, and is iterable
    for item in data:
        print(item)

# def update_one_data(collect,Id, data):   # this is used when we have to identity by automatic generated id by mongo
#
#    try:
#      obj_id =  ObjectId(Id)
#      collect.update_one(
#         {'_id': obj_id},
#         {'$set' : data}
#      )
#      f"record Updated successfully!!"
#    except Exception as e:
#        print(f"Error occurred!! {e}")



def update_one_data(collect, data, Id:int):
    """
    This function is used to update data. One data at a time!
    """
    try:
        if Id:
            collect.update_one(
                {'Id':Id},
                {'$set':data}
            )
        else:
            print(f"No Student found with this : {Id}")

    except Exception as e:
        print(f"Unexpected error!! {e}")



def update_many_data(collect, data):   # filter using Age & then update age
    """
    This function is used to update data. Multiple data at a time!
    """
    try:
        collect.update_many({'Name':"Om"},{'$set':data})

    except Exception as e:
        print(f"Unexpected error!! {e}")



def delete_data(collect, Id:int):
    """
    This function is used to delete data. One data at a time!
    """
    try:
        if Id:
            collect.delete_one({'_id':Id})
        else:
            print(f"No Student found with this : {Id}")

    except Exception as e:
        print(f"Unexpected error!! {e}")




if __name__ == '__main__':
    conn = connection_db()  # connecting pymongo to mongodb
    collection = create_client(conn, 'Student_db2', 'Students')  # creating db and returning collection
    #insert_one_data(collection,{'Name':'Kamlesh','Age':22,'Gender':'Male'})
    # insert_many_data(collection, [{'_id': 1 ,'Name':'Om','Age':22, 'Active':False},
    #                               {'_id':2,'Name':'Samarth','Age':22, 'City':'Atul'},
    #                               {'_id':3, 'Name':'Harsh','Age':21, 'City':'Valsad'},
    #                               {'_id':4, 'Name':'Alok', 'Age':21, 'City':'Pardi', 'Gender':'Male'},
    #                               {'_id':5, 'Name': 'kamlesh', 'Age': 22, 'City': 'Vapi'}
    #                               ])

    # update_one_data(collection,{'Name':'KamleshBhai'},5)
    # insert_one_data(collection, {"_id":1,"Name":"Om"})

    # update_many_data(collection, {'Age': 22)

    read_data(collection)

    delete_data(collection,4)

    read_data(collection)