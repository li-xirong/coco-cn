import sqlite3

from constant import DATABASE_FILE

conn = sqlite3.connect(DATABASE_FILE)

conn.execute('''CREATE TABLE IF NOT EXISTS IMAGESTATUE 
                (ID INT PRIMARY KEY   NOT NULL,
                IMAGEID         TEXT  NOT NULL,
                TIMES           INT   NOT NULL,
                LABEL           TEXT  NOT NULL);''')

conn.execute('''CREATE TABLE IF NOT EXISTS STATE
               (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                USER_ID             INT   NOT NULL,
                IMAGE_ID            INT   NOT NULL,
                SUBMIT_TIME         REAL  NOT NULL,
                SUGGESTED_SENTENCE  TEXT,
                RANK                INT,
                SUBMITTED_SENTENCE  TEXT  NOT NULL,
                SUBMITTED_LABEL     TEXT  NOT NULL,
                REAL_IMAGE_ID       INT   NOT NULL);''')

conn.execute('''CREATE TABLE IF NOT EXISTS USER
               (ID    INTEGER PRIMARY KEY   AUTOINCREMENT,
                USER_NAME         TEXT  NOT NULL,
                PASSWORD          TEXT  NOT NULL,
                IMAGE_POOL        TEXT,
                CURSOR            INTEGER);''')

conn.commit()
conn.close()


