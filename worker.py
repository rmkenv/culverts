"""
worker.py
RQ worker that processes queued culvert analysis jobs.
Run this in a separate process/container.

Usage:
  python worker.py
"""

import os
import logging
from redis import Redis
from rq import Worker, Queue, Connection

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("rq_worker")

REDIS_URL = os.getenv("RQ_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))

listen = ['culvert-jobs']

def main():
    redis_conn = Redis.from_url(REDIS_URL)
    with Connection(redis_conn):
        worker = Worker(map(Queue, listen))
        logger.info("Starting RQ worker")
        worker.work()

if __name__ == "__main__":
    main()
