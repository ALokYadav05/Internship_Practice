import pyodbc


try:
    conn_str = (
        "driver={ODBC Driver 17 for SQL Server};"
        "server=1R-77;"
        "database=interns;"
        "Trusted_Connection=yes;"
    )

    with pyodbc.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            query_insert = "INSERT INTO ai_intern (Name, Age, Email, City) VALUES (?, ?, ?, ?)"
            cursor.execute(query_insert, ("Raj", 21, "raj@gmail.com", "Pardi"))
            conn.commit()

            query_select = "SELECT * FROM ai_intern"
            cursor.execute(query_select)
            data = cursor.fetchall()

            for row in data:
                print(row)

            query_delete = "delete from ai_intern where Name = 'Raj'"
            conn.execute(query_delete)
            conn.commit()

            query_select = "SELECT * FROM ai_intern"
            cursor.execute(query_select)
            data = cursor.fetchall()

            for row in data:
                print(row)

except pyodbc.Error as e:
    print("Database error occurred:", e)
except Exception as ex:
    print("General error occurred:", ex)