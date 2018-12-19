from __future__ import print_function

# coding=utf-8
import os
import sys
import sqlite3

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
logger.setLevel(logging.INFO)

from constant import *
from simpleknn.simpleknn import load_model
from simpleknn.bigfile import BigFile

import image
import userControl
reload(sys)
sys.setdefaultencoding('utf8')

class sentence:
    rootpath = ROOT_PATH
    vis_feat = VIS_FEAT
    sent_collection = SENT_COLLECTION
    img_collection = IMG_COLLECTION

    sent_feat_dir = os.path.join(rootpath, sent_collection, "FeatureData", vis_feat)
    sent_id_file = os.path.join(sent_feat_dir, 'id.txt')
    shape_file = os.path.join(sent_feat_dir, 'shape.txt')
    sent_file = os.path.join(rootpath, sent_collection, 'TextData', '%s.txt' % sent_collection)

    def __init__(self, db_file):
        self.nr_of_sents, self.feat_dim = map(int, open(self.shape_file).readline().split())
        self.sent_pool = map(str.strip, open(self.sent_file).readlines())
        self.sent_searcher = load_model(os.path.join(self.sent_feat_dir, 'feature.bin'), self.feat_dim,
                                        self.nr_of_sents, self.sent_id_file)
        self.sent_searcher.set_distance('cosine')
        feat_dir = os.path.join(self.rootpath, self.img_collection, "FeatureData", self.vis_feat)
        self.vis_feat_file = BigFile(feat_dir)
        imageSetFile = open(os.path.join(self.rootpath, self.img_collection, "ImageSets", "%s.txt"%self.img_collection), 'r')
        self.imageSet = imageSetFile.readlines()
        self.db_file = db_file

    def getSentence(self, imageID):
        image = [self.imageSet[imageID].replace("\n", "")]
        renamed, vectors = self.vis_feat_file.read(image)

        result = []
        for i in range(len(renamed)):
            sent_list = self.sent_searcher.search_knn(vectors[i], max_hits=10)
            logger.info('query img', renamed[i])
            for sent_id, distance in sent_list[:5]:
                logger.info(self.sent_pool[int(sent_id[4:])].decode('utf-8'))
                result.extend([self.sent_pool[int(sent_id[4:])].decode('utf-8')])
            print ('')
        return result

    def save_sentence(self, user_id, image_id, submit_time, suggested_sentence, rank, 
                        submitted_sentence, labels, real_id):
        conn = sqlite3.connect(self.db_file)
        conn.text_factory = str
        cursor = conn.execute("SELECT user_id FROM STATE \
                WHERE user_id = %d AND image_id = %d" % (int(user_id), image_id))
        judge = -1
        for row in cursor:
            judge = row[0]
        if judge == -1:
            conn.execute("INSERT INTO STATE (USER_ID, IMAGE_ID, SUBMIT_TIME, SUGGESTED_SENTENCE, RANK, SUBMITTED_SENTENCE, SUBMITTED_LABEL, REAL_IMAGE_ID) \
                VALUES (%d, %d, %f, '%s', %d, '%s', '%s', %d)" % (
                user_id, image_id, submit_time, suggested_sentence.decode('gbk'), rank,
                submitted_sentence.decode('gbk'), labels, real_id))
        else:
            conn.execute("UPDATE STATE SET submit_time=%f, suggested_sentence='%s', rank=%d, submitted_sentence='%s', submitted_label='%s' \
                WHERE user_id = %d AND image_id = %d" % (
                submit_time, suggested_sentence.decode('gbk'), rank, submitted_sentence.decode('gbk'), labels,
                int(user_id),
                image_id))
        conn.commit()
        conn.close()

    def get_sentence(self, user_id, page):
        data = []
        img = image.image(self.db_file)
        conn = sqlite3.connect(self.db_file)
        conn.text_factory = str
        cursor = conn.execute("SELECT count(image_id) FROM STATE WHERE user_id=%d" % user_id)
        for row in cursor:
            count = row[0]
        if (page - 1) * PAGE_LIMIT > count:
            return False, None, None

        cursor = conn.execute("SELECT user_id, image_id, submit_time, suggested_sentence, rank, submitted_sentence, submitted_label FROM STATE \
            WHERE user_id = %d ORDER BY submit_time DESC LIMIT %d OFFSET %d" % (
            user_id, PAGE_LIMIT, (page - 1) * PAGE_LIMIT))

        import userControl as u
        user_control = u.user(self.db_file)

        for row in cursor:
            image_id = row[1]
            #import userControl as u
            #user_control = u.user(self.db_file)

            j, iid, image_id = user_control.getimageid(user_id, image_id)
            url = IMAGE_ROOT + img.getimagename(iid)
            submit_time = row[2]
            suggested_sentence = row[3]
            rank = row[4]
            submitted_sentence = row[5]
            submitted_label = row[6]

            set = {'image_id': image_id, 'url': url, 'submit_time': submit_time,
                   'suggested_sentence': suggested_sentence.encode('gbk'), 'rank': rank,
                   'submitted_sentence': submitted_sentence.encode('gbk'),
                   'submitted_label': submitted_label}

            data = data + [set]
        conn.close()
        import math
        return True, data, math.ceil(float(count) / PAGE_LIMIT)

    def get_sentence_by_imageid(self, user_id, image_id):
        conn = sqlite3.connect(self.db_file)
        conn.text_factory = str
        cursor = conn.execute("SELECT user_id, image_id, submit_time, suggested_sentence, rank, submitted_sentence, submitted_label FROM STATE \
                WHERE user_id = %d AND image_id = %d" % (int(user_id), image_id))
        count = 0
        for row in cursor:
            submitted_label = row[6]
            submitted_sentence = row[5]
            rank = row[4]
            count += 1

        conn.close()

        if count == 0:
            return 0, '', ''
        logger.info(submitted_label)
        return rank, submitted_sentence.encode('gbk'), ', '.join(
            filter(lambda x: x, submitted_label.split(', ')))

    def getNumber(self, user_id):
        conn = sqlite3.connect(self.db_file)
        conn.text_factory = str
        cursor = conn.execute("SELECT user_id, image_id, submit_time, suggested_sentence, rank, submitted_sentence FROM STATE \
            WHERE user_id = %d" % int(user_id))
        number = 0
        for row in cursor:
            number += 1
        return number

    def getAll(self, start, end):
        user_control = userControl.user(self.db_file)

        conn = sqlite3.connect(self.db_file)
        conn.text_factory = str
        cursor = conn.execute("SELECT user_id, count(image_id) FROM STATE WHERE SUBMIT_TIME < %f AND SUBMIT_TIME > %f \
                              GROUP BY user_id" % (end, start))
        logger.info("SELECT user_id, count(image_id) FROM STATE WHERE SUBMIT_TIME < %f AND SUBMIT_TIME > %f GROUP BY user_id" % (end, start))

        data = []
        for row in cursor:
            user_id = row[0]
            count = row[1]
            set = [user_id, count, user_control.getusername(user_id)]
            data = data + [set]
        return data

    def get_image_info(self, image_id):
        user_control = userControl.user(self.db_file)

        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute(
            "SELECT user_id, submitted_sentence, submitted_label FROM STATE WHERE real_image_id=%d" % image_id)
        data = []
        for row in cursor:
            user_id = row[0]
            submitted_sentence = row[1]
            submitted_label = row[2]
            set = [user_control.getusername(user_id), submitted_sentence, submitted_label]
            data = data + [set]

        return data
