from __future__ import annotations

import argparse

import uvicorn

from .db import initialize_database


def main() -> None:
    parser = argparse.ArgumentParser(description="作业/试卷批改 Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--init-db", action="store_true", help="仅初始化数据库")
    args = parser.parse_args()
    initialize_database()
    if not args.init_db:
        uvicorn.run("grading_agent.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
