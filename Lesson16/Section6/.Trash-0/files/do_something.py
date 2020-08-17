class Grader(object):
    
    def __init__(self, student_func):
        self.test_func = student_func
        
    def print_func(self):
        print(self.test_func)


if __name__ == "__main__":
    
    import numpy as np
    grader = Grader(np.mean)
    grader.print_func()
