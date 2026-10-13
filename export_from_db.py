import pandas as pd
import psycopg2

def export_data_from_postgres():
    connection = None
    try:
        # PostgreSQL bağlantısı (telco_churn_db ve şifre 1234)
        connection = psycopg2.connect(
            database="telco_churn_db",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )

        # SQL sorgusu ile verileri Pandas DataFrame'e çekme
        query = "SELECT * FROM telco_customers;"
        df = pd.read_sql(query, connection)

        # CSV dosyasına kaydetme
        output_filename = "Telco-Customer-Churn-FromDB.csv"
        df.to_csv(output_filename, index=False)
        print(f"Toplam {len(df)} satır veritabanından başarıyla çekildi ve '{output_filename}' olarak kaydedildi!")

    except Exception as error:
        print(f"Veri aktarım hatası: {error}")
    finally:
        if connection:
            connection.close()
            print("Veritabanı bağlantısı sonlandırıldı.")

if __name__ == "__main__":
    export_data_from_postgres()