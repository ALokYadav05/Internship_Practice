from pymongo import MongoClient
from  dotenv import load_dotenv
import os


load_dotenv()


class PymongoAdv:
    """
    initialize the connection , database and collections with pymongo
    """
    def __init__(self):
        self.url = os.getenv('MONGO_CONN_URL')
        self.client = MongoClient(self.url)
        self.db_name = self.client["interns"]
        self.collection_name = self.db_name["users"]

        print(f"Connected to {self.db_name}.{self.collection_name} successfully!")


    def read_data(self):
        """
        fetch data from database
        """
        rows = self.collection_name.find()
        if not rows:
            raise Exception('No data found!')
        for i in rows:
            print(i)
        return f" Data fetched Successfully!"


    def insert_many_data(self):
        """
        insert multiple data into database
        :return: the inserted data object
        """
        data = self.collection_name.insert_many([{'_id':1, 'Name':'Alok', 'Age':21, 'Marks': [34,45,29]},
                                                 {'_id':2, 'Name':'Kamlesh', 'Age':23, 'Marks': [44,35,20]},
                                                 {'_id': 10, 'Name': 'Samarth', 'Age': 21, 'Marks': [32, 39, 49]},
                                                 {'_id':3, 'Name':'Om', 'Age':22, 'Marks': [46,45,19]},
                                                 {'_id':4, 'Name':'Harsh', 'Age':22, 'Marks': [40,35,29]},
                                                 {'_id':6, 'Name':'Yash', 'Age':21, 'Marks': [44,32,19]}])
        return data


    def insert_one_data(self):
        """
        insert one data into database
        :return: the inserted data object
        """
        data = self.collection_name.insert_one({'_id':7, 'Name':'Rudra', 'Age':21,})

        return data


    def tried_diff_fetch_data(self):
        """
        Here, I have tried to fetch data from database using different technique
        :return:
        """
        r1 = self.collection_name.find({'Age': { '$in' : [22]}})
        r2 = self.collection_name.find({'Name' : {'$ne': 'Alok'}})
        r3 = self.collection_name.find({
                                      '$and' : [{'Age':21}, {'Name':'Alok'}]
                                      })

        r4 = self.collection_name.find({
                                      '$and' : [{'Age':21}, {'Name':'Alok'}]
                                    })

        r5 = self.collection_name.find({
                                     '$or' : [{'Age':21} , {'Age':22}]
                                 })

        r6 = self.collection_name.find({
                                     '$nor' : [{'Age':21}]
                                 })

        r7 = self.collection_name.find({
                                     'Marks' : {'$all': [40,29]}
                                 })

        r8 = self.collection_name.find({
                                    'Marks' : {
                                        '$elemMatch' : {'$gte':45}
                                    }
                                 })

        return r1, r2, r3, r4, r5, r6, r7, r8


    def pipeline_aggregate(self):
        """
        Here, I had used pipeline, to filter data and query and the same time
        :return: rows, which is object of collection
        """
        rows = self.collection_name.update_many(
                                     { '$expr':
                                           {'$gte': [{'$sum':'$Marks'},113]
                                            }
                                      },

                                     {'$set':
                                           {'Status':'Pass'}
                                    })
        return rows



    def update_many_data(self):
        """
        Here, I had updated multiple data into database using condition (if-else)
        :return: rows , which is object of collection
        """
        rows = self.collection_name.update_many({}, [{
                                                                '$set': {'Status':
                                                                    {
                                                                        '$cond': {
                                                                            'if': {'$lte': [{'$sum': '$Marks'}, 100]},
                                                                            'then': 'Pass',
                                                                            'else': 'Fail'
                                                                        }
                                                                    }
                                                                }
                                                            }]
                                                     )
        return rows



if __name__ == "__main__":
    conn = PymongoAdv()
    conn.read_data()
    # conn.update_many_data()
    # conn.insert_one_data()
    # conn.insert_many_data()
    # conn.pipeline_aggregate()
