import numpy as np

class Pcontrol_Sleep():
    def __init__(self, ratio, set_point, upper_bound, lower_bound):
        self.ratio = ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.set_point = set_point
    
    def next_per(self, cur_val):
        delta = ratio*(self.set_point-cur_val)
        next_val = cur_val + delta
        
        if next_val < self.lower_bound:
            next_val = self.lower_bound
        if next_val > self.upper_bound:
            next_val = self.upper_bound
            
        return next_val
    
    def set_slo(self, new_slo):
        self.set_point = new_slo
    
    def set_ratio(self, new_ratio):
        self.ratio = new_ratio


class Pcontrol_Util():
    def __init__(self, ratio, set_point, upper_bound, lower_bound):
        self.ratio = ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.set_point = set_point
    
    def next_per(self, cur_val):
        delta = ratio*(self.set_point-cur_val)
        next_val = cur_val + delta
        
        if next_val < self.lower_bound:
            next_val = self.lower_bound
        if next_val > self.upper_bound:
            next_val = self.upper_bound
            
        return next_val
    
    def set_slo(self, new_slo):
        self.set_point = new_slo
    
    def set_ratio(self, new_ratio):
        self.ratio = new_ratio
        