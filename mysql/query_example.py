import pymysql

localhost = ""
user = ""
password = ""
database = ""
charset = "utf8mb4"
table = "forecast"

#select station,datetime,PM25,AMB_TEMP,RAINFALL,RH,WIND_DIREC,WIND_SPEED from national_station_data where datetime>"2017-09-30" and station="二林";

connection = pymysql.connect(host=localhost, user=user, passwd=password, db=database, charset=charset)

try:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `%s`;"%(table)
        cursor.execute(sql)
        result = cursor.fetchone()
        print(result)

    '''
    with connection.cursor() as cursor:
        sql = "INSERT INTO `%s` (station,...) VALUES ("",...);"%(table)
        cursor.execute(sql)

    connection.commit()
    '''
except Exception as e:
    print("Error: ", e)

finally:
    connection.close()
