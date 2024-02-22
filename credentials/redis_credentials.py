import redis

def get_redis_object():
    redis_host = ''
    redis_password = ''
    Redis = redis.StrictRedis(host=redis_host, port=6380, password=redis_password, ssl=True)
    return Redis