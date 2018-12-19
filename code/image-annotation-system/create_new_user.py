import logging
import sqlite3

import userControl
from constant import DATABASE_FILE

logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)

class Users:

    def __init__(self, dbfile):
        self.dbfile = dbfile
        self.open_db()

    def open_db(self):
        self.conn = sqlite3.connect(self.dbfile)

    def close_db(self, dirty=True):
        if dirty:
            self.conn.commit()
        self.conn.close()

    def check_user(self, user_name):
        cursor = self.conn.execute("SELECT * FROM USER WHERE USER_NAME = '%s'" % user_name)
        if len(cursor.fetchall()) > 0:
            return 1
        return 0

    def check_passwd(self, password):
        if len(password) == 0:
            return 1
        return 0

    def add_user(self, user_name, password):
        if self.check_user(user_name):
            logging.info('%s already exists. Please ceate another user.', user_name)
            self.close_db(False)
            return 
        if self.check_passwd(password):
            logging.info('password is empty, please try again')
            self.close_db(False)
            return 

        user = userControl.user()
        pool = user.init_second_pool([[]])
        
        self.conn.execute("INSERT INTO USER (USER_NAME, PASSWORD, IMAGE_POOL, CURSOR) \
        VALUES ('%s', '%s', '%s', 0)" % (user_name, password, ', '.join(str(i) for i in pool[0])))

        self.close_db(True)


if __name__ == "__main__":
    user_name = raw_input('Please input user name: ')
    user_passwd = raw_input('Please input user passwd: ')
    Users(DATABASE_FILE).add_user(user_name, user_passwd)
