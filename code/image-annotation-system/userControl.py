import sqlite3
import random
import setup


class user:
    def __init__(self, db_file):
        self.db_file = db_file

    def checkUser(self, username, password):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT id, user_name, password FROM USER WHERE user_name='%s'" % username)
        m_username = None
        for row in cursor:
            id = row[0]
            m_username = row[1]
            m_password = row[2]

        cursor.close()
        conn.close()
        if m_username is None:
            return 2
        if password == m_password:
            return 1
        else:
            return 3

    def getUserId(self, username):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT id, user_name, password FROM USER WHERE user_name='%s'" % username)
        m_username = None
        for row in cursor:
            id = row[0]
            m_username = row[1]
            m_password = row[2]

        if m_username is None:
            raise Exception("No Such User")
        conn.close()
        return id

    @staticmethod
    def init_first_pool():
        pool = range(0, setup.MAX)
        first_pool = random.sample(pool, setup.FIRST_NUM)
        user_pool = []
        for i in range(0, setup.USER_NUM):
            user_pool = user_pool + [[]]
        user_group = range(0, setup.USER_NUM)
        for item in first_pool:
            user_sample = random.sample(user_group, 5)
            for i in user_sample:
                user_pool[i].extend([item])
        return user_pool

    @staticmethod
    def init_second_pool(user_pool):
        image_pool = range(0, setup.MAX)
        for pool in user_pool:
            left_pool = filter(lambda x: x not in pool, image_pool)
            add_pool = random.sample(left_pool, setup.NUM - len(pool))
            # print setup.NUM - len(pool)
            pool.extend(add_pool)
        return user_pool

    def init_image_pool(self):
        user_pool = self.init_second_pool(self.init_first_pool())

        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT * FROM USER")
        for i, row in enumerate(cursor):
            uid = row[0]
            pool = user_pool[i]
            conn.execute(
                "UPDATE USER SET image_pool='%s', cursor=0 WHERE id=%d" % (', '.join(str(i) for i in pool), uid))

        conn.commit()
        conn.close()

    def update_cursor(self, uid):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT * FROM USER WHERE id=%d" % uid)
        for row in cursor:
            c = row[4]

        conn.execute("UPDATE USER SET cursor=%d WHERE id=%d" % (c + 1, uid))
        conn.commit()
        conn.close()

    def getimageid(self, uid, iid):
        if iid >= setup.NUM:
            image_id = iid - 1
        else:
            image_id = iid
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT * FROM USER WHERE id=%d" % uid)
        for row in cursor:
            pool = row[3].split(", ")

        conn.close()
        if iid >= setup.NUM:
            return True, int(pool[image_id]), image_id
        else:
            return False, int(pool[image_id]), image_id

    def getusername(self, uid):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT user_name FROM USER WHERE id=%d" % uid)
        user_name = None
        for row in cursor:
            user_name = row[0]
        conn.close()
        return user_name

