# SuperFastPython.com
# example of handling an exception raised within a task
from time import sleep
from concurrent.futures import ThreadPoolExecutor
 
# mock task that will sleep for a moment
def work(value):
    sleep(1)
    try:
        raise Exception('Something bad happened!')
    except Exception as e:
        print("exception raised")
        raise e
 
# create a thread pool
with ThreadPoolExecutor() as executor:
    # executor.submit(work, 1)
    # result = executor.map(work, [1])
    for result in executor.map(work, [1]):
        print("hallo",result)
