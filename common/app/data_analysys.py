class Data:
    def __init__(self, data) -> None:
        self.data = data
        anal = Analitics(data)
        
    @property
    def get_analytics(self) -> None:
        return self.anal
    
class Analitics:
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return None
    
    def add_data(self, data)->None:
        self.data+=data
    
    def get_logs(self):
        return None
     