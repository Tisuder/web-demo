import abc
import numpy as np
from PIL import Image
import cv2

class RemoteModel(abc.ABC):
    @abc.abstractmethod
    def execute(self):
        pass



class RemoteVozModel(RemoteModel):
    def __init__(self, file: bytes) -> None:
        self.file = file
        self.img = self.image()
        
    def execute(self):
        # model = InternalModel()
        # model.foo()
        # model.bar()
        # return len(self.file) #Возвращаем результат анализа
        self.show()

    def image(self):
        name = '1.jpg'
        open(name, 'wb').write(self.file)
        # print(name)
        img = np.array(Image.open(name))
        return img
    
    def show(self, time:int=5*10**3):
        '''Create extra window with current image
        Args: time(msec)
        '''
        winname = 'image'
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, [750,500])
        cv2.imshow(winname, img)
        cv2.setWindowProperty(winname, cv2.WND_PROP_ASPECT_RATIO,
                      cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(time)
        cv2.destroyAllWindows()



class InternalModel():
    def foo(self):
        pass
    
    def bar(self):
        pass



def file_to_bytes(filename):
    return open(filename, 'rb').read()

# a = RemoteVozModel(file=file_to_bytes('1.jpg')).show()
