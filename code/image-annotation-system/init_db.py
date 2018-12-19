import os
import time
import logging
import sqlite3

from tqdm import tqdm

from constant import *


def init_imagestatus(conn):
    logging.info('Initializing IMAGESTATUS...')
    imset_file = os.path.join(ROOT_PATH, IMG_COLLECTION, "ImageSets","%s.txt"%IMG_COLLECTION)

    with  open(imset_file, 'r') as fr:
        for (num, value) in enumerate(fr.readlines()):
            conn.execute("INSERT INTO IMAGESTATUE (ID, IMAGEID, TIMES, LABEL) \
            VALUES (%d, '%s', 0, '')" % (num, value.strip()))

def init_user(conn):
    logging.info('Initializing USER...')
    conn.execute("INSERT INTO USER (USER_NAME, PASSWORD) \
        VALUES ('%s', '%s')" % ('test001', '2018'))


def update_imagestatus(conn):
    logging.info('Updating IMAGESTATUS...')
    recommended_label_file = os.path.join(ROOT_PATH, IMG_COLLECTION, "%s.tagvotes.txt"%IMG_COLLECTION)
 
    with open(recommended_label_file, 'r') as fr:
        for row in tqdm(fr.readlines()):
            labels = row.split()
            image_id = labels[0]
            label = "%s %s %s %s %s" % (labels[1], labels[3], labels[5], labels[7], labels[9])
            conn.execute("UPDATE IMAGESTATUE SET label='%s' \
                        WHERE imageid = '%s'" % (label, image_id))

if __name__ == '__main__':
    since = time.time()

    conn = sqlite3.connect(DATABASE_FILE)
    init_imagestatus(conn)
    update_imagestatus(conn)
    conn.commit()
    conn.close()

    elapse_time = time.time()-since
    logging.info('Initializing Database Finished in %dm %ds', elapse_time // 60, elapse_time % 60)
