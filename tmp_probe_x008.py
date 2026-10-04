from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from lab import runner


SCENES = [f"T0-dev-{i:02d}" for i in range(1, 7)] + [
    f"T1-dev-{i:02d}" for i in range(1, 7)
]


if __name__ == "__main__":
    gated = runner.gates("v3_grasp", SCENES, False)
    args = (gated["scenes"][0], gated["cfg"], 10_000, None, False)
    print("gate", gated["cfg"]["stages"])
    print("direct_smoke", runner._episode(args))
    for method in ("spawn", "fork"):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=get_context(method)
        ) as executor:
            print(method, executor.submit(runner._episode, args).result())
