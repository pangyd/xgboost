# encoding:utf-8

import os
import multiprocessing
import time

start_time = time.time()
def aqi(i):
    os.system("python ./{}.py".format(i))


t1 = multiprocessing.Process(target=aqi, args=(1,))
t2 = multiprocessing.Process(target=aqi, args=(2,))
t3 = multiprocessing.Process(target=aqi, args=(3,))
t4 = multiprocessing.Process(target=aqi, args=(4,))
t5 = multiprocessing.Process(target=aqi, args=(5,))
# t6 = multiprocessing.Process(target=aqi, args=(6,))
# t7 = multiprocessing.Process(target=aqi, args=(7,))
# t8 = multiprocessing.Process(target=aqi, args=(8,))
# t9 = multiprocessing.Process(target=aqi, args=(9,))
# t10 = multiprocessing.Process(target=aqi, args=(10,))
# t11 = multiprocessing.Process(target=aqi, args=(1,))
# t12 = multiprocessing.Process(target=aqi, args=(2,))
# t13 = multiprocessing.Process(target=aqi, args=(3,))
# t14 = multiprocessing.Process(target=aqi, args=(4,))
# t15 = multiprocessing.Process(target=aqi, args=(5,))
# t16 = multiprocessing.Process(target=aqi, args=(6,))
# t17 = multiprocessing.Process(target=aqi, args=(7,))
# t18 = multiprocessing.Process(target=aqi, args=(8,))
# t19 = multiprocessing.Process(target=aqi, args=(9,))
# t20 = multiprocessing.Process(target=aqi, args=(10,))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
# t6.start()
# t7.start()
# t8.start()
# t9.start()
# t10.start()
# t11.start()
# t12.start()
# t13.start()
# t14.start()
# t15.start()
# t16.start()
# t17.start()
# t18.start()
# t19.start()
# t20.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
# t6.join()
# t7.join()
# t8.join()
# t9.join()
# t10.join()
# t11.join()
# t12.join()
# t13.join()
# t14.join()
# t15.join()
# t16.join()
# t17.join()
# t18.join()
# t19.join()
# t20.join()

end_time = time.time()
print(end_time - start_time)
#
#
# from multiprocessing import Pool, cpu_count
#
# p = Pool(6)
# ret = p.map(aqi, [1, 2, 3])  # 自带close,join
# p.close()  # 关闭进程池，不再接受新的进程
# p.join()  # 主进程阻塞等待子进程结束
# t = time.time()
# print(t-start_time)

