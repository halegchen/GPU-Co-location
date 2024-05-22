class tic_svc:
    def __init__(self,cur_time=0,delta=1):
        self.delta = delta # clock accuracy in sec
        self.cur_time = cur_time
        
    def tic_job(self):
        self.cur_time = self.cur_time + self.delta
        
    def get_cur_time(self):
        return self.cur_time
    
