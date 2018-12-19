# coding=utf-8
from __future__ import print_function
import os
import sys
import sqlite3

import image
from constant import DATABASE_FILE

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
logger.setLevel(logging.INFO)


reload(sys)
sys.setdefaultencoding('utf8')


def process(options):
    dbfile = options.dbfile
    overwrite = options.overwrite
    resfile = os.path.splitext(dbfile)[0] + '.txt'
    
    if overwrite:
        if os.path.exists(resfile):
            logger.info('%s exists. overwrite', resfile)
    else:
        if os.path.exists(resfile):
            logger.info('%s exists. skip', resfile)   
            return

    img = image.image(dbfile)
    conn = sqlite3.connect(dbfile)
    conn.text_factory = str

    cursor = conn.execute("SELECT user_id, image_id, submit_time, suggested_sentence, rank, submitted_sentence, submitted_label, real_image_id FROM STATE")

    fw = open(resfile, 'w')

    img_list = []
    user_list = []
    for row in cursor:
        user_list.append(row[0])
        #image_id = row[7] #img.getimagename(row[1])
        image_id = img.getimagename(row[1])
        submit_time = row[2]
        suggested_sentence = row[3]
        rank = row[4]
        submitted_sentence = row[5]
        submitted_label = row[6]
        fw.write('%s\t%s\t%s\t%s\n' % (image_id, row[0], submitted_sentence, submitted_label))
        img_list.append(image_id)
    fw.close()

    logger.info('Number of annotations: %d', len(img_list))
    logger.info('Number of images: %d', len(set(img_list)))
    logger.info('Number of users: %d', len(set(user_list)))



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--dbfile", default=DATABASE_FILE, type="string", help="database file (default: %s)"%DATABASE_FILE)

    (options, args) = parser.parse_args(argv)

    return process(options)


if __name__=="__main__":
    sys.exit(main())
