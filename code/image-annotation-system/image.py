from __future__ import print_function
import sqlite3
import random


class image:
    def __init__(self, dbfile):
        self.dbfile = dbfile
        pool = []
        conn = sqlite3.connect(self.dbfile)
        cursor = conn.execute("SELECT ID FROM IMAGESTATUE WHERE TIMES < 3")
        for row in cursor:
            pool.extend([row[0]])
        self.pool = pool
        conn.close()

    def getimageid(self, uid):
        conn = sqlite3.connect(self.dbfile)
        cursor = conn.execute("SELECT * FROM USER WHERE id=%d" % uid)
        for row in cursor:
            c = row[4]
        conn.close()
        return c

    def getimagename(self, iid):
        conn = sqlite3.connect(self.dbfile)

        cursor = conn.execute("SELECT imageid FROM IMAGESTATUE WHERE ID = %d" % iid)
        for row in cursor:
            name = row[0]
        conn.close()
        return name

    def getimageidbyname(self, iname):
        conn = sqlite3.connect(self.dbfile)
        cursor = conn.execute("SELECT ID FROM IMAGESTATUE WHERE imageid = '%s'" % iname)
        iid = -1
        for row in cursor:
            iid = row[0]
        conn.close()
        return int(iid)

    def getlabel(self, id):
        conn = sqlite3.connect(self.dbfile)
        cursor = conn.execute("SELECT label FROM IMAGESTATUE WHERE ID = %d" % id)
        for row in cursor:
            label = row[0]
        conn.close()
        return label

    def save(self, id):
        conn = sqlite3.connect(self.dbfile)
        cursor = conn.execute("SELECT times FROM IMAGESTATUE WHERE ID = %d" % id)
        for row in cursor:
            time = row[0]
        conn.execute("UPDATE IMAGESTATUE SET times=%d WHERE ID=%d" % (time + 1, id))
        conn.commit()
        conn.close()

if __name__ == '__main__':
    from constant import DATABASE_FILE
    img = image(DATABASE_FILE)
    print (img.getimagename(1))
