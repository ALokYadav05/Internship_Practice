from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base


try:
    engine = create_engine("mssql+pyodbc://localhost/interns?driver=ODBC+Driver+17+for+SQL+Server",echo=True)
    print("Connection successful")
except SQLAlchemyError as e:
    print("Error while connecting to database", e)


base = declarative_base()  # defining ORM model

class Person(base):
    __tablename__ = "ai_intern"
    id = Column(Integer, primary_key=True , autoincrement=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)

Session = sessionmaker(bind=engine)
session = Session()

# to insert one row
# try :
#     p1 = Person(name="Abhishek",age=21)
#     session.add(p1)
#     session.commit()
#     print("Person added successfully")
# except SQLAlchemyError as e:
#     session.rollback()
#     print("Adding to db failed", e)
#
# # to insert multiple rows
#
# try:
#     new_person = [Person(name='Raju',age=34),
#                   Person(name='Harshit',age=35),
#                   Person(name='Nain',age=35)
#                   ]
#     session.add_all(new_person)
#     session.commit()
#     print("Persons added successfully")
# except SQLAlchemyError as e:
#     session.rollback()
#     print("Adding multiple records failed", e)

users = session.query(Person).all()
for user in users:
    print(user)



session.close()




# def get_db():
#     db = Session()
#     try:
#         yield db
#     except ConnectionError as e:
#         print(e)
#     except Exception as e:
#         print(e)
#     finally:
#         db.close()

