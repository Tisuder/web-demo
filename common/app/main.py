import hashlib
import os
import random
import zipfile
from fastapi import FastAPI, File, Path, Request, status, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated, Any, Callable, TypeVar

from app.models import *


nums = list(range(10000))
T = TypeVar('T')

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')




@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/upload', response_class=HTMLResponse)
async def upload(request: Request, file: Annotated[bytes, File()]):
    '''
    Сделать тут обработку картинки
    
    '''
    if (validate_file(file)):
        try:
            remote = RemoteVozModel(file) #Вызываем класс обрабатывающий картинку
            result = remote.execute()

            return templates.TemplateResponse('index.html',
                                              {'request': request, 'output': result, 'success': 1})
        except Exception as ex:
            print(ex)
            return templates.TemplateResponse('index.html', {'request': request, 'output': 'Oops, some errors was occurred', 'success': 0})

    else:
        return templates.TemplateResponse('index.html',
                                          {'request': request, 'output': 'Only .zip|.rar files are allowed', 'success': 0})


def validate_file(file: bytes): return 1
    # rand_name = hashlib.md5(str(random.choice(nums)).encode()).hexdigest()
    # open(rand_name, 'wb').write(file)

    # print(rand_name)
    # zipped = zipfile.is_zipfile(rand_name)
    # os.remove(rand_name)
    # return zipped

# def first(src: list[T], cond: Callable[[T], bool]) -> T | None:
#     for obj in src:
#         if cond(obj):
#             return obj
#     return None
