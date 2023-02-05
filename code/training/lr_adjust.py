import numpy as np

def current_lr(relative_progress, interval,lr0, lr1):
    return lr0 + (1-np.cos(np.pi * relative_progress/interval))*(lr1-lr0)/2


#template
def lr_adjust(current_progress, progress_list=[], lr_list=[]):
    l = len(progress_list)
    for i in range(l-1):
        if current_progress > progress_list[i] and current_progress < progress_list[i+1]:
            relative_progress = current_progress-progress_list[i]
            interval = progress_list[i+1]-progress_list[i]
            return current_lr(relative_progress, interval, lr_list[i], lr_list[i+1])

