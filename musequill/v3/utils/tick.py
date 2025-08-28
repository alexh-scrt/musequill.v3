import time
import logging
from .time_utils import seconds_to_time_string

logger = logging.getLogger(__name__)

def tick(start:float, end:float) -> str:
    if start > end:
        logger.warning(f"⚠️ Warning - start:{start} > end:{end}")
    secs = end - start
    lapse = seconds_to_time_string(secs)
    logger.info(f"⏱️ Took {lapse} or {end - start:.6f} sec")
    return lapse

if __name__ == "__main__":
    start = time.perf_counter()
    # your function or block
    end = time.perf_counter()

    print(f"Took {end - start:.6f} seconds")
