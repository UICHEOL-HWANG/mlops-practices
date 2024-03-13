import psycopg2 

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS mnist_images(
            id SERIAL PRIMARY KEY,
            image_data BYTEA,
            label INTEGER
    )
    
    """
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="postgres-server",
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect)
