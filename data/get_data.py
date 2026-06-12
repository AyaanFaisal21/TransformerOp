"""Download the Tiny Shakespeare dataset (~1.1 MB of text).

Small enough to train on in minutes, structured enough that a working model
produces recognizably Shakespeare-shaped output -- which makes bugs obvious.
"""

from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).parent / "shakespeare.txt"


def main() -> None:
    if OUT.exists():
        print(f"already downloaded: {OUT} ({OUT.stat().st_size:,} bytes)")
        return
    print(f"downloading {URL} ...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    OUT.write_text(resp.text, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
