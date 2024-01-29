import time


class TimeLogging:

    @staticmethod
    def log_time(start_time, title):
        curr_time = time.time()

        if start_time is None:
            print(title)
            return curr_time

        print(f'total time for {title}:{int(curr_time - start_time)}s')
        return curr_time
