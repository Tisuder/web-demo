import abc


class RemoteModel(abc.ABC):

    @abc.abstractmethod
    def execute(self):
        pass


class RemoteKoblikModel(RemoteModel):
    def __init__(self, file: bytes) -> None:
        self.file = file

    def execute(self):
        model = InternalModel()
        model.foo()
        model.bar()

        return len(self.file)


class InternalModel():
    def foo(self):
        pass

    def bar(self):
        pass
