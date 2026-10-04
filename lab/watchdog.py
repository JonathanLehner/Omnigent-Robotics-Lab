"""Lab watchdog: wake the PI when the lab is idle (no busy agent, no new record) for IDLE_MIN minutes.

A sub-agent that dies silently never reports to the PI's inbox, so the PI waits forever. This nudges it.
  uv run python -m lab.watchdog <session_id>     (started by ./lab.sh run)
"""

import json
import sys
import time
import urllib.request

from lab import record

SERVER = "http://127.0.0.1:6767"
IDLE_MIN = 20


def _get(path):
    with urllib.request.urlopen(SERVER + path, timeout=15) as r:
        return json.load(r)


def _busy(session_id):
    pi = _get(f"/v1/sessions/{session_id}")
    children = _get(f"/v1/sessions/{session_id}/child_sessions?limit=200")
    items = children.get("data") or children.get("sessions") or []
    return bool(pi.get("busy")) or any(c.get("busy") for c in items)


def _nudge(session_id, idle_min):
    msg = (f"From the lab watchdog: nothing has happened for {idle_min} minutes and no agent is busy. A sub-agent may "
           "have died without reporting (its result will never arrive). Check the state with query_record and the "
           "git branches, re-dispatch any task that did not finish, and continue autonomously.")
    body = json.dumps({"type": "message", "data": {"role": "user", "content": [{"type": "input_text", "text": msg}]}})
    req = urllib.request.Request(f"{SERVER}/v1/sessions/{session_id}/events", data=body.encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=15)


def main(session_id):
    last_change, last_count = time.time(), None
    while True:
        try:
            count = len(record.query(limit=100000))
            if count != last_count:
                last_change, last_count = time.time(), count
            idle = (time.time() - last_change) / 60
            if idle >= IDLE_MIN and not _busy(session_id):
                _nudge(session_id, int(idle))
                last_change = time.time()  # give the PI a full window before nudging again
        except Exception as e:  # noqa: BLE001 - the watchdog must outlive server restarts
            print("watchdog:", e, file=sys.stderr)
        time.sleep(120)


if __name__ == "__main__":
    main(sys.argv[1])
