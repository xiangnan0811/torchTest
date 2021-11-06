import time
from tqdm import tqdm

a = 1000
b = range(a)
c = tqdm(b, total=a)
for i in c:
    time.sleep(0.1)

