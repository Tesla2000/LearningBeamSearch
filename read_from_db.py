import sqlite3

import numpy as np


def main():
    # Connect to the SQLite database
    conn = sqlite3.connect('data_3_25.db')

    # Create a cursor object
    cur = conn.cursor()

    # Execute a query to select data from the table
    cur.execute('SELECT * FROM Samples')

    # Fetch all rows from the result set
    rows = cur.fetchall()

    # Iterate over the rows and print each row
    for row in rows:
        print(np.array(tuple(int.from_bytes(value, byteorder='little') for value in row[1:])))

    # Close the connection
    conn.close()


if __name__ == '__main__':
    main()
