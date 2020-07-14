import sqlite3

conn = sqlite3.connect("test.db")

cursor =conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(itemno INTEGER, Category TEXT,
                FoodName TEXT, Company TEXT, price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)
                
#데이터 넣자
sql = "INSERT into supermarket(Itemno, Category, Foodname, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))

sql = "INSERT into supermarket(Itemno, Category, Foodname, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (2, '음료수', '코코넛드링크', '편의점', 2000))

sql = "INSERT into supermarket(Itemno, Category, Foodname, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (33, '고기', '한우', '하나로마트', 10000))

sql = "INSERT into supermarket(Itemno, Category, Foodname, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (4, '박카스', '약', '약국', 1000))

sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno, Categoty, Foodname, Company, Price FROM supermarket"

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[3]) + " " +
              str(row[3]) + " " + str(row[4]))

conn.commit()
conn.close




