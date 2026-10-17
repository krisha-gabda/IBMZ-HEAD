"""
db_utils.py - read learner data from, and write risk scores back to, the database.

Production backend : IBM Db2 on IBM Z (LinuxONE) via the `ibm_db` driver
Local test backend : SQLite (same code path, no Db2 needed)

Configuration is read from environment variables, so no passwords live in the repo:

    DB_BACKEND        db2 (default) | sqlite
    DB2_HOST          e.g. 192.168.1.10
    DB2_PORT          default 50000
    DB2_DATABASE      e.g. NEETDB
    DB2_USER
    DB2_PASSWORD
    DB2_SCHEMA        optional, e.g. NEET
    DB2_INPUT_TABLE   default NEET_DATA
    DB2_OUTPUT_TABLE  default NEET_RISK_SCORES
    DB_SQLITE_PATH    default data/neet_local.db   (sqlite backend only)

Install on LinuxONE:  pip install ibm_db
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_TABLE = os.getenv("DB2_INPUT_TABLE", "NEET_DATA")
OUTPUT_TABLE = os.getenv("DB2_OUTPUT_TABLE", "NEET_RISK_SCORES")

# Result table layout (must match ibm_z/db2_create_table.sql)
OUTPUT_DDL_COLUMNS = """
    learner_id             INTEGER,
    risk_score             DOUBLE,
    risk_level             VARCHAR(10),
    top_factor_1           VARCHAR(100),
    top_factor_2           VARCHAR(100),
    top_factor_3           VARCHAR(100),
    suggested_intervention VARCHAR(255)
"""
OUTPUT_COLUMNS = [
    "learner_id", "risk_score", "risk_level",
    "top_factor_1", "top_factor_2", "top_factor_3", "suggested_intervention",
]


def _qualified(table):
    schema = os.getenv("DB2_SCHEMA")
    return f"{schema}.{table}" if schema else table


def get_connection():
    backend = os.getenv("DB_BACKEND", "db2").lower()
    if backend == "sqlite":
        import sqlite3

        path = os.getenv("DB_SQLITE_PATH", str(ROOT / "data" / "neet_local.db"))
        return sqlite3.connect(path)

    try:
        import ibm_db_dbi
    except ImportError as e:
        raise ImportError("ibm_db is not installed. Run: pip install ibm_db") from e

    needed = ["DB2_HOST", "DB2_DATABASE", "DB2_USER", "DB2_PASSWORD"]
    missing = [v for v in needed if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing environment variables for Db2: {missing}")

    conn_str = (
        f"DATABASE={os.environ['DB2_DATABASE']};"
        f"HOSTNAME={os.environ['DB2_HOST']};"
        f"PORT={os.getenv('DB2_PORT', '50000')};"
        "PROTOCOL=TCPIP;"
        f"UID={os.environ['DB2_USER']};"
        f"PWD={os.environ['DB2_PASSWORD']};"
    )
    if os.getenv("DB2_SSL", "").lower() in ("1", "true", "yes"):
        conn_str += "SECURITY=SSL;"
    return ibm_db_dbi.connect(conn_str, "", "")


def read_learners(table=None):
    """SELECT * from the learner table. Column names are lower-cased (Db2 upper-cases them)."""
    table = _qualified(table or INPUT_TABLE)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        cols = [d[0].lower() for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()
    # Db2 DECIMAL columns come back as Decimal objects - make numerics real numbers
    for c in df.columns:
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() == df[c].notna().sum():
                df[c] = converted
    return df


def _py(v):
    """numpy / NaN -> plain Python types the DB driver accepts."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def write_scores(result, table=None):
    """Replace the contents of the output table with `result` (full refresh)."""
    table = _qualified(table or OUTPUT_TABLE)
    rows = [
        tuple(_py(v) for v in row)
        for row in result[OUTPUT_COLUMNS].astype(object).itertuples(index=False, name=None)
    ]
    conn = get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT 1 FROM {table} WHERE 1=0")
        except Exception:
            conn.rollback()
            cur.execute(f"CREATE TABLE {table} ({OUTPUT_DDL_COLUMNS})")
            conn.commit()
        cur.execute(f"DELETE FROM {table}")
        placeholders = ",".join("?" * len(OUTPUT_COLUMNS))
        cur.executemany(
            f"INSERT INTO {table} ({','.join(OUTPUT_COLUMNS)}) VALUES ({placeholders})", rows
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def read_scores(table=None):
    """Read the risk-score table (used by the dashboard)."""
    return read_learners(table or OUTPUT_TABLE)
