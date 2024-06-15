import lmdb


env = lmdb.open("/home/huyixiao/DataCache/sad", readonly=True, lock=False, readahead=False,
                    meminit=False)

txn = env.begin()
cur = txn.cursor()
for key,value in cur:
    print(key)