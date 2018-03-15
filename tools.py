import time

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        duration = time.time()-self.start_time;
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.duration = time.time()-self.start_time;
        self.start()
        return self.duration

if __name__ == '__main__':
    # Test usages

    # To start timer
    timer = Timer()

    # reset timer
    timer.tic()

    # show time usage, donot reset
    timer.show("Time usages: ")

    # To obtain timming and reset
    duration = timer.end()

    # To obtain timming reset time and output information
    timer.toc("Runing time: ")
