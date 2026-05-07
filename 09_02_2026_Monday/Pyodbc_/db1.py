import pyodbc

print(pyodbc.drivers())

conn_str = "driver={ODBC Driver 17 for SQL Server}; server={1R-77}; database={interns}; Trusted_Connection=yes;"

conn = pyodbc.connect(conn_str)

cursor = conn.cursor()

query = "select * from ai_intern"
query2 = "insert into ai_intern (Name,Age,Email,City) values ('Rudra',22,'rudra@gmail.com','pardi')"
cursor.execute(query2)
cursor.execute(query)
data = cursor.fetchall()

for row in data:
    print(row)

cursor.close()
conn.close()

