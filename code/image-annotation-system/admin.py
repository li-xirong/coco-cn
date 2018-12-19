from __future__ import print_function

import web
import constant
import sentence
import userControl
import time
import image

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
logger.setLevel(logging.INFO)

from setup import START
from constant import *

web.config.debug = False

render = web.template.render('adminTemplates/')
urls = (
    '/', 'index',
    '/image', 'image',
    '/history', 'history',
    '/single', 'single'
)

sen = sentence.sentence(DATABASE_FILE)
user_control = userControl.user(DATABASE_FILE)
image_pool = image.image(DATABASE_FILE)


class index:
    def GET(self):
        i = web.input()
        start = i.get("start")
        end = i.get("end")

        if start is None:
            start = START
        if end is None:
            end = time.time()

        data = sen.getAll(float(start), float(end))
        resp = {"data": data}
        return render.index(resp)

    def POST(self):
        i = web.input()
        start_mark = int(i.get("start_mark"))

        end_mark = int(i.get("end_mark"))

        if start_mark == 1:
            start_year = int(i.get("start_year"))
            start_month = int(i.get("start_month"))
            start_date = int(i.get("start_date"))
            start_hour = int(i.get("start_hour"))
            start_minute = int(i.get("start_minute"))
            start_t = (start_year, start_month, start_date, start_hour, start_minute, 0, 0, 0, 0)
            start_time = time.mktime(start_t)
        else:
            start_time = START

        if end_mark == 1:
            end_year = int(i.get("end_year"))
            end_month = int(i.get("end_month"))
            end_date = int(i.get("end_date"))
            end_hour = int(i.get("end_hour"))
            end_minute = int(i.get("end_minute"))
            end_t = (end_year, end_month, end_date, end_hour, end_minute, 0, 0, 0, 0)
            end_time = time.mktime(end_t)
        else:
            end_time = time.time()
        return start_time, end_time


class history:
    def GET(self):
        i = web.input()
        user = int(i.get("user"))

        i = web.input()
        page = i.get("page")
        if page is None:
            page = 1

        judge, data, page_number = sen.get_sentence(user, int(page))
        resp = {'judge': judge, 'data': data, 'page': page_number, 'current': page, 'user': user}
        return render.history(resp)


class image:
    def GET(self):
        i = web.input()
        user = int(i.get("user"))
        image_id = int(i.get("image_id"))
        rank, s_sen, s_label = sen.get_sentence_by_imageid(user, image_id)
        j, imageid, image_id = user_control.getimageid(user, image_id)
        image_name = image_pool.getimagename(imageid)
        label = ', '.join(filter(lambda x: x, image_pool.getlabel(imageid).split(' ')))
        logger.info(label)

        image_url = constant.IMAGE_ROOT + image_name

        sentences = sen.getSentence(imageid)
        logger.info("s_sen: %s", s_sen)

        number = sen.getNumber(user)

        logger.info("s_label: %s", s_label)

        resp = {'image_id': image_id, 'image': image_url, 'sentence': sentences, 'label': label,
                'submitted_sentence': s_sen, 'submitted_label': s_label, 'rank': rank, 'number': number,
                'judge': j, 'user': user}
        return render.image(resp)

class single:
    def GET(self):
        return render.image_own()

    def POST(self):
        i = web.input()
        image_name = i.get("image")
        image_id = image_pool.getimageidbyname(image_name)
        data = sen.get_image_info(image_id)
        label = ', '.join(filter(lambda x: x, image_pool.getlabel(image_id).split(' ')))
        image_url = constant.IMAGE_ROOT + image_name
        sentences = sen.getSentence(image_id)
        resp = {'image_id': image_id, 'image_url': image_url, 'label': label, 'sentence': sentences, 'data': data}
        return render.image_own(resp)

app = web.application(urls, globals())

if __name__ == "__main__":
    app.run()
