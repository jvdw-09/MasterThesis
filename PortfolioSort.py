from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import polars as pl
import gc
import time
from datetime import timedelta
from tqdm import tqdm
from arch.bootstrap import StationaryBootstrap

# --- Paths ---
DATA_POS   = Path("/Users/jortvanderweijde/Desktop/Pythontest/Positions")
TIMEFRAMES = ["15 hour position"]
TF_TO_UNIT = {"15MINUTE": "MINUTE"}
DB_PATH    = "/Users/jortvanderweijde/Desktop/Pythontest/crypto.duckdb"
RISKFREE   = Path("/Users/jortvanderweijde/Desktop/Pythontest/Riskfree_fixed.csv")
RESULTS_DIR = Path("/Users/jortvanderweijde/Desktop/Pythontest/Results")

# --- Setup DB ---
def setup_database(con):
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA max_temp_directory_size='80GB'")
    con.execute("""
        CREATE TABLE IF NOT EXISTS pos_all (
            ts TIMESTAMP,
            symbol VARCHAR,
            timeframe VARCHAR,
            rule_id VARCHAR,
            position SMALLINT
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ret_all (
            ts TIMESTAMP,
            symbol VARCHAR,
            timeframe VARCHAR,
            log_ret DOUBLE
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_ret (
            date DATE,
            name  VARCHAR,
            log_ret DOUBLE
        );
    """)

# --- File Loader ---
def csv_gz_to_parquet_with_ts(con, path: Path) -> Path:
    """Create a parquet copy of the gz CSV with identical columns + parsed ts, once."""
    # Peek header to get column names
    df_sample = con.execute(f"""
        SELECT * FROM read_csv_auto('{path.as_posix()}', compression='gzip') LIMIT 5
    """).fetchdf()
    cols = df_sample.columns
    ts_col = cols[0]

    # Write a parquet that includes the parsed ts (exact same parsing logic you used)
    pq_path = path.with_suffix("").with_suffix(".parquet")  # turns .csv.gz -> .parquet
    con.execute(f"""
        COPY (
            SELECT
                *,
                CASE
                    WHEN length(trim(CAST("{ts_col}" AS VARCHAR))) = 10 THEN
                        CAST(trim(CAST("{ts_col}" AS VARCHAR)) || ' 00:00:00' AS TIMESTAMP)
                    ELSE
                        CAST(trim(CAST("{ts_col}" AS VARCHAR)) AS TIMESTAMP)
                END AS ts
            FROM read_csv_auto('{path.as_posix()}', compression='gzip')
        )
        TO '{pq_path.as_posix()}' (FORMAT 'parquet');
    """)
    return pq_path

def insert_ret_all_from_parquet(con, pq_path: Path, symbol: str, tf: str, price_col: str):
    """Exact same ret_all, but shifted one earlier (use lead), drop last row."""
    con.execute(f"""
        INSERT INTO ret_all
        SELECT
            ts,                                  -- keep current timestamp
            '{symbol}' AS symbol,
            '{tf}' AS timeframe,
            ln(lead("{price_col}") OVER (ORDER BY ts)) - ln("{price_col}") AS log_ret
        FROM read_parquet('{pq_path.as_posix()}')
        WHERE "{price_col}" IS NOT NULL
        QUALIFY log_ret IS NOT NULL             -- drops the final row where lead() is NULL
    """)

def insert_pos_all_from_parquet_batched(con, pq_path: Path, symbol: str, tf: str, rule_cols, batch_size: int = 64):
    """Memory-friendly melt in column batches; identical long-format result."""
    # Fetch ts + a batch of rule columns each time to keep RAM bounded
    for i in range(0, len(rule_cols), batch_size):
        batch = rule_cols[i:i+batch_size]
        quoted = [f'"{c}"' for c in batch]
        sql = f"""
            SELECT ts, {", ".join(quoted)}
            FROM read_parquet('{pq_path.as_posix()}')
        """
        df_raw = con.execute(sql).fetchdf()

        melted_frames = []
        for col in batch:
            m = pd.DataFrame({
                'ts': df_raw['ts'],
                'symbol': symbol,
                'timeframe': tf,
                'rule_id': f"{symbol}_{col}",     # exact same rule_id pattern
                'position': df_raw[col]
            })
            m.dropna(subset=['position'], inplace=True)
            m['position'] = m['position'].astype(int)
            melted_frames.append(m)

        if melted_frames:
            df_melted = pd.concat(melted_frames, ignore_index=True)
            con.register("df_pos_batch", df_melted)
            con.execute("INSERT INTO pos_all SELECT * FROM df_pos_batch;")
            con.unregister("df_pos_batch")
            del df_melted, melted_frames, df_raw
            gc.collect()

# --- Main file loader ---
def melt_file_to_duck(con, path: Path, seen_price: set):
    base = path.stem
    parts = base.split('_')
    symbol, tf = parts[1], parts[2]

    # Get column names once
    df_sample = con.execute(f"""
        SELECT * FROM read_csv_auto('{path.as_posix()}', compression='gzip') LIMIT 5
    """).fetchdf()
    all_cols = df_sample.columns
    ts_col = all_cols[0]
    price_col = all_cols[1]
    rule_cols = all_cols[2:] 

    # 1) Convert to parquet (with ts) once; subsequent steps read parquet
    pq_path = csv_gz_to_parquet_with_ts(con, path)

    # 2) ret_all
    if (symbol, tf) not in seen_price:
        insert_ret_all_from_parquet(con, pq_path, symbol, tf, price_col)
        seen_price.add((symbol, tf))

    # 3) pos_all via batched melt
    insert_pos_all_from_parquet_batched(con, pq_path, symbol, tf, rule_cols, batch_size=64)

# --- Riskfree loader ---
def load_riskfree(con, riskfree_path: Path):
    con.execute(f"""
        CREATE OR REPLACE TABLE rf_daily AS
        SELECT * FROM read_csv_auto('{riskfree_path}');
    """)

# --- Holding period ---
def apply_holding_period_sql(con,
                             periods: int,
                             unit: str,
                             bar_size: int,
                             symbols,
                             batch_rules_non_excl: int = 300,
                             batch_rules_excl: int = 200,
                             chunk_months_excl: int = 6):
    """
    Holding-period expansion using events (+ at start, − at first grid after inclusive end)
    and cumulative sums on the grid. Branching:
      - Non-exclusive symbols: rule-batched (fast path)
      - Exclusive symbols (BTC, ETH, LTC, XRP, BCH): rule-batched + time-chunked

    At the end, use add_zero_rules_batched(...) to add back rules that were always zero.
    """
    unit = unit.upper()
    if unit not in ("MINUTE", "HOUR", "DAY"):
        raise ValueError("unit must be 'MINUTE', 'HOUR' or 'DAY'")

    extend = periods - 1
    totalPeriod = extend * bar_size  # number of base bars inside holding window
    print(f"→ Holding {periods} bars ({bar_size} {unit}) — branching for symExclusive")

    r_pattern      = "(MA|FR|SR|RSI|CH)"
    secondRule_pat = "d[0-5]"
    thirdRule_pat  = "[jvp]"
    symExclusive   = {"BCHUSD", "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"}

    # Target table
    con.execute("DROP TABLE IF EXISTS pos_all_extended;")
    con.execute("""
        CREATE TABLE pos_all_extended(
            ts TIMESTAMP,
            symbol VARCHAR,
            timeframe VARCHAR,
            rule_id VARCHAR,
            position SMALLINT
        );
    """)

    # Remember which rules were ALWAYS zero in the original pos_all
    con.execute("""
        CREATE OR REPLACE TEMP VIEW zero_rules AS
        WITH universe AS (
            SELECT DISTINCT symbol, timeframe, rule_id
            FROM pos_all
        ),
        nonzero AS (
            SELECT DISTINCT symbol, timeframe, rule_id
            FROM pos_all
            WHERE position <> 0
        )
        SELECT u.symbol, u.timeframe, u.rule_id
        FROM universe u
        LEFT JOIN nonzero nz USING(symbol, timeframe, rule_id)
        WHERE nz.rule_id IS NULL;
    """)

    for sym in symbols:
        print(f"  • {sym}")

        # Keep only non-zero starts
        con.execute(f"""
            CREATE OR REPLACE TABLE nz AS
            SELECT DISTINCT ts, symbol, timeframe, rule_id, position
            FROM pos_all
            WHERE symbol = '{sym}' AND position <> 0;
        """)
        if con.sql("SELECT COUNT(*) FROM nz;").fetchone()[0] == 0:
            con.execute("DROP TABLE IF EXISTS nz;")
            continue

        # Filter rules
        if sym in symExclusive:
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW nz_filt AS
                SELECT ts, symbol, timeframe, rule_id, position
                FROM nz
                WHERE regexp_matches(rule_id, '.*({r_pattern}).*({thirdRule_pat}).*({secondRule_pat}).*');
            """)
        else:
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW nz_filt AS
                SELECT ts, symbol, timeframe, rule_id, position
                FROM nz
                WHERE regexp_matches(rule_id, '.*({r_pattern}).*')
                  AND regexp_matches(rule_id, '.*({secondRule_pat}).*');
            """)

        if con.sql("SELECT COUNT(*) FROM nz_filt;").fetchone()[0] == 0:
            con.execute("DROP VIEW IF EXISTS nz_filt;")
            con.execute("DROP TABLE IF EXISTS nz;")
            continue

        # Time bounds (for this symbol)
        t0, t1 = con.execute("SELECT MIN(ts), MAX(ts) FROM nz_filt;").fetchone()
        t0_ts = pd.Timestamp(t0)
        t1_ts = pd.Timestamp(t1)

        # ---------- BRANCH ----------
        if sym in symExclusive:
            # Exclusive: rule-batched + time-chunked
            rule_ids = [r[0] for r in con.sql("SELECT DISTINCT rule_id FROM nz_filt;").fetchall()]
            if not rule_ids:
                con.execute("DROP VIEW IF EXISTS nz_filt;")
                con.execute("DROP TABLE IF EXISTS nz;")
                continue

            cur_ts = t0_ts.to_period("M").to_timestamp()
            end_all_ts = t1_ts

            while cur_ts <= end_all_ts:
                chunk_end_ts = cur_ts + pd.DateOffset(months=chunk_months_excl) - pd.Timedelta(days=1)
                if chunk_end_ts > end_all_ts:
                    chunk_end_ts = end_all_ts

                cur_s = cur_ts.strftime('%Y-%m-%d %H:%M:%S')
                end_s = chunk_end_ts.strftime('%Y-%m-%d %H:%M:%S')
                t0_s  = t0_ts.strftime('%Y-%m-%d %H:%M:%S')
                t1_s  = t1_ts.strftime('%Y-%m-%d %H:%M:%S')

                print(f"    · chunk {cur_s} → {end_s}")

                con.execute(f"""
                    CREATE OR REPLACE TABLE gs AS
                    SELECT g.generate_series
                    FROM generate_series(
                           TIMESTAMP '{t0_s}',
                           TIMESTAMP '{t1_s}',
                           INTERVAL '{bar_size} {unit}'
                         ) AS g
                    WHERE g.generate_series BETWEEN TIMESTAMP '{cur_s}' AND TIMESTAMP '{end_s}';
                """)

                for i in range(0, len(rule_ids), batch_rules_excl):
                    batch = rule_ids[i:i + batch_rules_excl]
                    in_list = ",".join(f"'{r}'" for r in batch)

                    con.execute(f"""
                        CREATE OR REPLACE TEMP VIEW nz_sub AS
                        SELECT * FROM nz_filt
                        WHERE rule_id IN ({in_list})
                          AND ts <= TIMESTAMP '{end_s}'
                          AND ts + INTERVAL '{totalPeriod} {unit}' >= TIMESTAMP '{cur_s}';
                    """)
                    if con.sql("SELECT COUNT(*) FROM nz_sub;").fetchone()[0] == 0:
                        con.execute("DROP VIEW IF EXISTS nz_sub;")
                        continue

                    con.execute(f"""
                        CREATE OR REPLACE TEMP VIEW events AS
                        SELECT symbol, timeframe, rule_id, ts AS ts_event, position AS delta
                        FROM nz_sub
                        UNION ALL
                        SELECT symbol, timeframe, rule_id,
                               ts + INTERVAL '{totalPeriod + bar_size} {unit}' AS ts_event,
                               -position AS delta
                        FROM nz_sub;
                    """)

                    con.execute("""
                        CREATE OR REPLACE TEMP VIEW ev_agg AS
                        SELECT e.symbol, e.timeframe, e.rule_id, e.ts_event AS ts, SUM(e.delta) AS delta
                        FROM events e
                        JOIN gs ON gs.generate_series = e.ts_event
                        GROUP BY 1,2,3,4;
                    """)

                    con.execute(f"""
                        WITH grid_rules AS (
                            SELECT r.symbol, r.timeframe, r.rule_id, g.generate_series AS ts
                            FROM (SELECT DISTINCT symbol, timeframe, rule_id FROM nz_sub) r
                            JOIN gs g ON TRUE
                        ),
                        cum_deltas AS (
                            SELECT
                                gr.ts, gr.symbol, gr.timeframe, gr.rule_id,
                                COALESCE(b.base, 0)
                                + SUM(COALESCE(e.delta, 0)) OVER (
                                    PARTITION BY gr.symbol, gr.timeframe, gr.rule_id
                                    ORDER BY gr.ts
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                                ) AS cum
                            FROM grid_rules gr
                            LEFT JOIN ev_agg e
                              ON e.symbol=gr.symbol
                             AND e.timeframe=gr.timeframe
                             AND e.rule_id=gr.rule_id
                             AND e.ts=gr.ts
                            LEFT JOIN (
                                -- base cumulative before the chunk (grid-aligned)
                                SELECT e.symbol, e.timeframe, e.rule_id, SUM(e.delta) AS base
                                FROM events e
                                JOIN (
                                  SELECT g.generate_series AS gts
                                  FROM generate_series(
                                         TIMESTAMP '{t0_s}',
                                         TIMESTAMP '{cur_s}' - INTERVAL '{bar_size} {unit}',
                                         INTERVAL '{bar_size} {unit}'
                                       ) AS g
                                ) g ON g.gts = e.ts_event
                                GROUP BY 1,2,3
                            ) b
                              ON b.symbol=gr.symbol AND b.timeframe=gr.timeframe AND b.rule_id=gr.rule_id
                        )
                        INSERT INTO pos_all_extended
                        SELECT
                            ts, symbol, timeframe, rule_id,
                            CASE WHEN cum > 0 THEN 1
                                 WHEN cum < 0 THEN -1
                                 ELSE 0 END AS position
                        FROM cum_deltas;
                    """)

                    con.execute("DROP VIEW IF EXISTS ev_agg;")
                    con.execute("DROP VIEW IF EXISTS nz_sub;")

                cur_ts = cur_ts + pd.DateOffset(months=chunk_months_excl)
                con.execute("DROP TABLE IF EXISTS gs;")

            con.execute("DROP VIEW IF EXISTS nz_filt;")
            con.execute("DROP TABLE IF EXISTS nz;")

        else:
            # Non-exclusive: rule-batched across full span
            t0_s = t0_ts.strftime('%Y-%m-%d %H:%M:%S')
            t1_s = t1_ts.strftime('%Y-%m-%d %H:%M:%S')

            con.execute(f"""
                CREATE OR REPLACE TABLE gs AS
                SELECT g.generate_series
                FROM generate_series(
                       TIMESTAMP '{t0_s}',
                       TIMESTAMP '{t1_s}',
                       INTERVAL '{bar_size} {unit}'
                     ) AS g;
            """)

            rule_ids = [r[0] for r in con.sql("SELECT DISTINCT rule_id FROM nz_filt;").fetchall()]
            for i in range(0, len(rule_ids), batch_rules_non_excl):
                batch = rule_ids[i:i + batch_rules_non_excl]
                in_list = ",".join(f"'{r}'" for r in batch)

                con.execute(f"CREATE OR REPLACE TEMP VIEW nz_sub AS SELECT * FROM nz_filt WHERE rule_id IN ({in_list});")

                con.execute(f"""
                    CREATE OR REPLACE TEMP VIEW events AS
                    SELECT symbol, timeframe, rule_id, ts AS ts_event, position AS delta
                    FROM nz_sub
                    UNION ALL
                    SELECT symbol, timeframe, rule_id,
                           ts + INTERVAL '{totalPeriod + bar_size} {unit}' AS ts_event,
                           -position AS delta
                    FROM nz_sub;
                """)

                con.execute("""
                    CREATE OR REPLACE TEMP VIEW ev_agg AS
                    SELECT e.symbol, e.timeframe, e.rule_id, e.ts_event AS ts, SUM(e.delta) AS delta
                    FROM events e
                    JOIN gs ON gs.generate_series = e.ts_event
                    GROUP BY 1,2,3,4;
                """)

                con.execute("""
                    WITH grid_rules AS (
                        SELECT r.symbol, r.timeframe, r.rule_id, g.generate_series AS ts
                        FROM (SELECT DISTINCT symbol, timeframe, rule_id FROM nz_sub) r
                        JOIN gs g ON TRUE
                    ),
                    cum_deltas AS (
                        SELECT
                            gr.ts, gr.symbol, gr.timeframe, gr.rule_id,
                            SUM(COALESCE(e.delta, 0)) OVER (
                                PARTITION BY gr.symbol, gr.timeframe, gr.rule_id
                                ORDER BY gr.ts
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            ) AS cum
                        FROM grid_rules gr
                        LEFT JOIN ev_agg e
                          ON e.symbol=gr.symbol
                         AND e.timeframe=gr.timeframe
                         AND e.rule_id=gr.rule_id
                         AND e.ts=gr.ts
                    )
                    INSERT INTO pos_all_extended
                    SELECT
                        ts, symbol, timeframe, rule_id,
                        CASE WHEN cum > 0 THEN 1
                             WHEN cum < 0 THEN -1
                             ELSE 0 END AS position
                    FROM cum_deltas;
                """)

                con.execute("DROP VIEW IF EXISTS ev_agg;")
                con.execute("DROP VIEW IF EXISTS events;")
                con.execute("DROP VIEW IF EXISTS nz_sub;")

            con.execute("DROP TABLE IF EXISTS gs;")
            con.execute("DROP VIEW IF EXISTS nz_filt;")
            con.execute("DROP TABLE IF EXISTS nz;")

    print("✓ Holding period expansion done (non-zero rules). Now adding always-zero rules in batches…")

def add_zero_rules_batched(con, chunk_months: int = 6, threads: int = 4):
    """
    Add back rules that were ALWAYS zero in the original pos_all, in batches:
    per (symbol,timeframe) and per time chunk using ret_all's grid.
    Anti-join ensures no duplicates; memory stays bounded.
    """
    con.execute(f"PRAGMA threads={threads};")
    con.execute("PRAGMA preserve_insertion_order=false;")

    pairs = con.execute("""
        SELECT DISTINCT symbol, timeframe
        FROM zero_rules
    """).fetchall()

    for sym, tf in pairs:
        # ts range from ret_all
        t0, t1 = con.execute("""
            SELECT MIN(ts), MAX(ts)
            FROM ret_all
            WHERE symbol = ? AND timeframe = ?
        """, [sym, tf]).fetchone()
        if t0 is None or t1 is None:
            continue

        cur = pd.Timestamp(t0).to_period("M").to_timestamp()
        end = pd.Timestamp(t1)

        while cur <= end:
            chunk_end = cur + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1)
            if chunk_end > end:
                chunk_end = end

            cur_s = cur.strftime("%Y-%m-%d %H:%M:%S")
            end_s = chunk_end.strftime("%Y-%m-%d %H:%M:%S")

            con.execute("""
                INSERT INTO pos_all_extended
                SELECT
                    r.ts,
                    z.symbol,
                    z.timeframe,
                    z.rule_id,
                    0 AS position
                FROM (
                    SELECT ts
                    FROM ret_all
                    WHERE symbol = ? AND timeframe = ?
                      AND ts BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
                    ORDER BY ts
                ) r
                JOIN (
                    SELECT symbol, timeframe, rule_id
                    FROM zero_rules
                    WHERE symbol = ? AND timeframe = ?
                ) z ON TRUE
                LEFT JOIN pos_all_extended e
                  ON e.ts = r.ts
                 AND e.symbol = z.symbol
                 AND e.timeframe = z.timeframe
                 AND e.rule_id = z.rule_id
                WHERE e.ts IS NULL;
            """, [sym, tf, cur_s, end_s, sym, tf])

            cur = cur + pd.DateOffset(months=chunk_months)

# --- Create rule_ret table using Polars ---
def create_rule_ret_sql(con, chunk_months: int = 6):
    print("→ Building rule_ret in DuckDB (set-based)...")

    # Bars-per-day mapping
    con.execute("""
        CREATE OR REPLACE TEMP VIEW tf_factor AS
        WITH t AS (
            SELECT DISTINCT timeframe FROM pos_all
        )
        SELECT
            timeframe,
            CASE
                WHEN lower(timeframe) LIKE '%24hours%'  THEN 1
                WHEN lower(timeframe) LIKE '%12hours%'  THEN 2
                WHEN lower(timeframe) LIKE '%4hours%'   THEN 6
                WHEN lower(timeframe) LIKE '%1hour%'    THEN 24
                WHEN lower(timeframe) LIKE '%30minute%' THEN 48
                WHEN lower(timeframe) LIKE '%15minute%' THEN 96
                WHEN lower(timeframe) LIKE '%10minute%' THEN 144
                WHEN lower(timeframe) LIKE '%5minute%'  THEN 288
                ELSE 1
            END AS factor
        FROM t;
    """)

    # Precreate helper views
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ret AS
        SELECT ts, symbol, timeframe, log_ret AS r_b
        FROM ret_all;
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW rf AS
        SELECT CAST(date AS DATE) AS date, log_ret AS rf_day
        FROM rf_daily;
    """)

    # Target table
    con.execute("""
        CREATE OR REPLACE TABLE rule_ret (
            ts TIMESTAMP,
            symbol VARCHAR,
            timeframe VARCHAR,
            rule_id VARCHAR,
            D SMALLINT,
            r_b DOUBLE,
            r_rf DOUBLE,
            r_j DOUBLE
        );
    """)

    pairs = con.sql("SELECT DISTINCT symbol, timeframe FROM pos_all").fetchall()
    total = len(pairs)
    print(f"→ Processing {total} symbol–timeframe pairs (chunk={chunk_months} months)…")

    start_all = time.perf_counter()

    for idx, (sym, tf) in enumerate(pairs, start=1):
        t0_pair = time.perf_counter()

        # Quick size hint for logging
        n_pos = con.execute(
            "SELECT COUNT(*) FROM pos_all WHERE symbol=? AND timeframe=?",
            [sym, tf]
        ).fetchone()[0]

        # Time bounds for this pair
        t0, t1 = con.execute(
            "SELECT MIN(ts), MAX(ts) FROM pos_all WHERE symbol=? AND timeframe=?",
            [sym, tf]
        ).fetchone()

        if t0 is None or t1 is None:
            print(f"  [{idx:>3}/{total}] {sym} — {tf} | no rows, skipping")
            continue

        t0_ts = pd.Timestamp(t0)
        t1_ts = pd.Timestamp(t1)

        # Walk the range in N-month chunks
        cur_ts = t0_ts.to_period("M").to_timestamp()  # first day of that month
        inserted_rows = 0

        while cur_ts <= t1_ts:
            chunk_end_ts = cur_ts + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1)
            if chunk_end_ts > t1_ts:
                chunk_end_ts = t1_ts

            cur_s = cur_ts.strftime("%Y-%m-%d %H:%M:%S")
            end_s = chunk_end_ts.strftime("%Y-%m-%d %H:%M:%S")

            # Insert this time-chunk for the pair
            con.execute("""
                INSERT INTO rule_ret
                WITH pos AS (
                    SELECT
                        ts,
                        symbol,
                        timeframe,
                        rule_id,
                        CAST(position AS SMALLINT) AS D,
                        DATE(ts) AS date
                    FROM pos_all
                    WHERE symbol = ? AND timeframe = ?
                      AND ts BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
                )
                SELECT
                    p.ts,
                    p.symbol,
                    p.timeframe,
                    p.rule_id,
                    p.D,
                    CAST(r.r_b AS DOUBLE) AS r_b,
                    CAST(rf.rf_day / f.factor AS DOUBLE) AS r_rf,
                    CAST(CASE WHEN p.D = 0
                         THEN rf.rf_day / f.factor
                         ELSE p.D * r.r_b
                    END AS DOUBLE) AS r_j
                FROM pos p
                LEFT JOIN ret r
                       ON r.ts = p.ts
                      AND r.symbol = p.symbol
                      AND r.timeframe = p.timeframe
                LEFT JOIN rf
                       ON rf.date = p.date
                LEFT JOIN tf_factor f
                       ON f.timeframe = p.timeframe;
            """, [sym, tf, cur_s, end_s])

            # next chunk
            cur_ts = cur_ts + pd.DateOffset(months=chunk_months)

        dt_pair = time.perf_counter() - t0_pair
        dt_total = time.perf_counter() - start_all
        print(f"  [{idx:>3}/{total}] {sym} — {tf} | pos rows: {n_pos:,} | pair {timedelta(seconds=int(dt_pair))} | total {timedelta(seconds=int(dt_total))}")

    print("✅ rule_ret built (batched by pair + time chunks).")

# --- Aggregate daily returns ---
def aggregate_daily_returns_polars_batched(con, batch_size: int = 500):
    """
    Batch‑aggregate rule_ret → daily_ret in DuckDB, using Polars for each batch.
    Splits on rule_id to keep memory use bounded.
    """
    # (re)create the target table
    con.execute("""
        CREATE OR REPLACE TABLE daily_ret (
            d     DATE,
            rule_id VARCHAR,
            r_j_d DOUBLE,
            r_b_d DOUBLE
        );
    """)

    # grab all unique rule_ids
    rule_ids = con.sql("SELECT DISTINCT rule_id FROM rule_ret").df()["rule_id"].tolist()
    total = len(rule_ids)
    print(f"→ Aggregating {total} rules in batches of {batch_size}.")

    for i in tqdm(range(0, total, batch_size), desc="Daily agg batches"):
        batch = rule_ids[i : i + batch_size]
        # make SQL IN‑list, properly quoted
        quoted = ",".join(f"'{r}'" for r in batch)

        # fetch only this batch from DuckDB via Arrow
        arrow_tbl = con.execute(f"""
            SELECT 
              ts, 
              rule_id, 
              r_j AS r_j_d, 
              r_b AS r_b_d
            FROM rule_ret
            WHERE rule_id IN ({quoted})
        """).arrow()

        # turn into Polars, aggregate by day & rule_id
        pl_df = (
            pl.from_arrow(arrow_tbl)
              .with_columns(pl.col("ts").dt.date().alias("d"))
              .group_by(["d", "rule_id"])
              .agg([
                  pl.col("r_j_d").sum().alias("r_j_d"),
                  pl.col("r_b_d").sum().alias("r_b_d")
              ])
        )

        # push back into DuckDB
        con.register("tmp_daily", pl_df.to_arrow())
        con.execute("INSERT INTO daily_ret SELECT * FROM tmp_daily;")
        con.unregister("tmp_daily")

    # sanity check
    nulls = con.sql("SELECT COUNT(*) FROM daily_ret WHERE r_j_d IS NULL OR r_b_d IS NULL").fetchone()[0]
    print(f"→ Finished. NULL rows in daily_ret: {nulls}")

# --- Create performance tables ---
def create_performance_table(con):
    con.execute("""
        CREATE OR REPLACE TABLE perf_pooled AS
        SELECT 
            REGEXP_EXTRACT(rule_id, '[^_]+_(.*)', 1) AS core_rule,
            AVG(r_j_d - r_b_d) AS f_j,
            (AVG(r_j_d) / NULLIF(STDDEV_POP(r_j_d), 0)) - 
            (AVG(r_b_d) / NULLIF(STDDEV_POP(r_b_d), 0)) AS SR_j
        FROM daily_ret
        GROUP BY core_rule;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE perf_per_asset AS
        SELECT 
            rule_id,
            AVG(r_j_d - r_b_d) AS f_j,
            (AVG(r_j_d) / NULLIF(STDDEV_POP(r_j_d), 0)) - 
            (AVG(r_b_d) / NULLIF(STDDEV_POP(r_b_d), 0)) AS SR_j
        FROM daily_ret
        GROUP BY rule_id;
    """)
    print("✅ Performance tables created.")

# --- Aggregate coins into rule
def create_daily_returns_clean(con, symbols, batch_size):
    """
    Build daily_ret_clean only for the tickers in `symbols`, in batches
    to keep memory low.
    """
    # 1) Prep target table
    con.execute("""
        CREATE OR REPLACE TABLE daily_ret_clean (
            d DATE,
            clean_rule_id VARCHAR,
            r_j_d DOUBLE,
            r_b_d DOUBLE
        );
    """)

    # 2) Helper: temp view with prefix extracted once
    con.execute("""
        CREATE OR REPLACE TEMP VIEW daily_ret_prefixed AS
        SELECT
            d,
            regexp_extract(rule_id, '^([^_]+)')   AS coin,
            regexp_replace(rule_id, '^[^_]+_', '') AS clean_rule_id,
            r_j_d,
            r_b_d
        FROM daily_ret;
    """)

    # 3) Loop over symbol chunks
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        in_list = ",".join(f"'{s}'" for s in batch)

        con.execute(f"""
            INSERT INTO daily_ret_clean
            SELECT
                d,
                clean_rule_id,
                SUM(r_j_d) AS r_j_d,
                SUM(r_b_d) AS r_b_d
            FROM daily_ret_prefixed
            WHERE coin IN ({in_list})
            GROUP BY 1,2;
        """)
        print(f"  → inserted batch {i//batch_size+1}: {len(batch)} coins")

    # 4) Stats
    rows, rules = con.sql("""
        SELECT COUNT(*), COUNT(DISTINCT clean_rule_id) FROM daily_ret_clean
    """).fetchone()
    print(f"✅ daily_ret_clean ready — {rules} strategies, {rows} rows.")

# --- Calculate monthly p-value (identical method; faster via .apply) ---
def calculate_monthly_pvalues_batched(con,
                                      bootstrap_samples: int = 999,
                                      batch_size: int = 500):
    """
    Compute monthly‑frequency p‑values per strategy rule in batches.
    For each rule and each month (after the first 12),
    uses the prior 12 months of r_j_m to compute a stationary‑bootstrap
    t‑stat and p‑value (3‑month blocks), writing into p_values_monthly.
    """
    # 1) (Re)build monthly_ret
    con.execute("""
        CREATE OR REPLACE TABLE monthly_ret AS
        SELECT
            CAST(date_trunc('month', d) AS DATE) AS month,
            regexp_replace(rule_id, '^[^_]+_', '') AS clean_rule_id,
            SUM(r_j_d) AS r_j_m
        FROM daily_ret
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)

    # 2) list all months and rules
    months = [r[0] for r in con.sql("SELECT DISTINCT month FROM monthly_ret ORDER BY month").fetchall()]
    rules  = [r[0] for r in con.sql("SELECT DISTINCT clean_rule_id FROM monthly_ret").fetchall()]

    # 3) prepare output table
    con.execute("DROP TABLE IF EXISTS p_values_monthly")
    con.execute("""
        CREATE TABLE p_values_monthly (
            month         DATE,
            clean_rule_id VARCHAR,
            t             DOUBLE,
            p             DOUBLE
        );
    """)

    print(f"→ {len(rules)} rules to process in batches of {batch_size}")

    # 4) process in batches
    for start in tqdm(range(0, len(rules), batch_size), desc="Rule batches"):
        batch = rules[start : start + batch_size]
        in_list = ",".join(f"'{r}'" for r in batch)

        # pull this batch's data
        df = con.sql(f"""
            SELECT clean_rule_id, month, r_j_m
            FROM monthly_ret
            WHERE clean_rule_id IN ({in_list})
        """).df()

        # pivot to months × rules
        wide = df.pivot(index="month", columns="clean_rule_id", values="r_j_m")

        results = []
        for rule in batch:
            arr = wide[rule].values  # shape=(#months,)
            # slide a 12‐month window
            for i in range(12, len(months)):
                M      = months[i]
                window = arr[i-12:i]

                # skip if any NaN or zero variance
                if np.isnan(window).any() or np.std(window, ddof=1) == 0:
                    results.append((M, rule, np.nan, np.nan))
                    continue

                # original t‑stat
                t_orig = window.mean() / np.std(window, ddof=1)

                # stationary bootstrap with 3‑month blocks
                sb = StationaryBootstrap(3, window)
                count = 0
                for draw in sb.bootstrap(bootstrap_samples):
                    sample = draw[0]
                    # if still nested tuple, unpack again:
                    if isinstance(sample, tuple):
                        sample = sample[0]
                    sample = np.asarray(sample)
                    std = np.std(sample, ddof=1)
                    t_stat = sample.mean() / std if std > 0 else 0.0
                    if abs(t_stat) >= abs(t_orig):
                        count += 1

                pval = count / bootstrap_samples
                results.append((M, rule, float(t_orig), float(pval)))

        # write batch back
        tmp = pd.DataFrame(results, columns=["month","clean_rule_id","t","p"])
        con.register("tmp_pvals", tmp)
        con.execute("INSERT INTO p_values_monthly SELECT * FROM tmp_pvals")
        con.unregister("tmp_pvals")

    print("✅ Finished writing `p_values_monthly`")

# --- FDR selection ---
def fdr_selection_polars_monthly(con):
    # 1) Pull in the monthly p‐values
    arrow_tbl = con.execute("SELECT month, clean_rule_id, p FROM p_values_monthly WHERE p != 0").arrow()
    pl_df = pl.from_arrow(arrow_tbl).drop_nulls()

    # 2) Compute within‐month rank (k) and count (m)
    base = pl_df.with_columns([
        pl.col("p")
          .rank(method="ordinal")
          .over("month")
          .alias("k"),
        pl.count()
          .over("month")
          .alias("m")
    ])

    # 3) build one FDR block per level and concat
    fdr_frames = []
    for level in (5, 10, 20):
        α = level / 100.0
        fdr = base.with_columns([
            pl.lit(level).alias("fdr_level"),
            (pl.col("p") <= α * pl.col("k") / pl.col("m")).alias("is_selected")
        ])
        fdr_frames.append(fdr)

    selected = pl.concat(fdr_frames)

    # 4a) register a temporary view
    con.register("tmp_fdr", selected.to_arrow())

    # 4b) persist as a real DuckDB table
    con.execute("""
        CREATE OR REPLACE TABLE selected_rules_multi AS
        SELECT * FROM tmp_fdr;
    """)

    # 4c) drop the temporary view
    con.unregister("tmp_fdr")

    print("✅ Monthly FDR selection complete (5%, 10%, 20)")

# --- Rolling portfolios (with progress bar) ---
def rolling_portfolios_streamlined(con,
                                   window_months: int = 12,
                                   batch_size: int = 250):
    # 1) pull all calendar‐months
    months = [r[0] for r in con.sql(
        "SELECT DISTINCT date_trunc('month', d) AS month "
        "FROM daily_ret_clean "
        "ORDER BY month"
    ).fetchall()]
    
    rows = []
    perf_pooled = con.sql("SELECT * FROM perf_pooled").df().set_index("core_rule")["f_j"]
    
    for M in tqdm(months[window_months:], desc="Rolling months"):
        # IS / OOS windows
        end_is   = pd.to_datetime(M) - pd.offsets.MonthBegin(0)
        begin_is = (end_is - pd.offsets.DateOffset(months=window_months)) + pd.Timedelta(days=1)
        oos_start = M
        oos_end   = pd.to_datetime(M) + pd.offsets.MonthEnd(0)
        
        # 2) FDR winners _for this month_ by clean_rule_id
        fdr_dict = {}
        for lvl in (5,10,20):
            sel = con.sql(f"""
                SELECT clean_rule_id
                  FROM selected_rules_multi
                 WHERE month      = '{M}'
                   AND fdr_level = {lvl}
                   AND is_selected
            """).df()["clean_rule_id"].tolist()
            fdr_dict[lvl] = set(sel)
        
        # 3) who’s “active” in‐sample?
        active = con.sql(f"""
            SELECT clean_rule_id
              FROM daily_ret_clean
             WHERE d BETWEEN '{begin_is.date()}' AND '{end_is.date()}'
             GROUP BY clean_rule_id
            HAVING COUNT(*) >= {int(window_months*21)}
        """).df()["clean_rule_id"].tolist()
        
        # 4) pick top50/flop50 by pooled f_j
        sub = perf_pooled.reindex([r for r in active if r in perf_pooled.index]).dropna()
        top50, flop50 = sub.nlargest(50).index.tolist(), sub.nsmallest(50).index.tolist()
        
        # 5) build portfolios (all keys are clean_rule_id now)
        portfolios = {
          "long_1"      : top50[:1],
          "short_1"     : flop50[:1],
          "long_50"     : top50,
          "short_50"    : flop50,
          "longshort_50": top50+flop50,
        }
        for lvl in (5,10,20):
            plus  = [r for r in sub.index if r in fdr_dict[lvl] and sub[r] > 0]
            minus = [r for r in sub.index if r in fdr_dict[lvl] and sub[r] < 0]
            portfolios[f"long_FDR{lvl}"]      = plus
            portfolios[f"short_FDR{lvl}"]     = minus
            portfolios[f"longshort_FDR{lvl}"] = plus + minus
        
        # 6) for each portfolio, grab its daily returns
        for name, rules in portfolios.items():
            if not rules: 
                continue
            for i in range(0, len(rules), batch_size):
                batch = rules[i:i+batch_size]
                in_list = ",".join(f"'{r}'" for r in batch)
                df = con.sql(f"""
                    SELECT d    AS date,
                           AVG(r_j_d) AS port_r
                      FROM daily_ret_clean
                     WHERE d BETWEEN '{oos_start}' AND '{oos_end}'
                       AND clean_rule_id IN ({in_list})
                     GROUP BY d
                     ORDER BY d
                """).df()
                for _, r in df.iterrows():
                    rows.append((r["date"], name, r["port_r"]))                   
    
    con.execute("""
        CREATE OR REPLACE TABLE portfolio_ret (
            date DATE,
            name  VARCHAR,
            log_ret DOUBLE
        );
    """)
    # 7) bulk‐insert & pivot out
    con.executemany(
      "INSERT INTO portfolio_ret VALUES (?,?,?)",
      rows
    )
    df = con.sql("SELECT date, name, log_ret FROM portfolio_ret").df()
    wide = df.pivot(
        index=["date"],
        columns="name",
        values="log_ret"
    ).reset_index()
    wide.to_csv(RESULTS_DIR/"portfolio_returns_wide.csv", index=False)
    print("✅ Done.")

# --- Entry Point ---
def main():
    symbolsTop = ["AAVEUSD", "ADAUSD", "ALGOUSD", "AVAXUSD", "AXSUSD", "BCHUSD", "BTCUSD",
                  "CRVUSD", "DAIUSD", "ETHUSD", "FETUSD", "FTMUSD", "GALAUSD", "GRTUSD",
                  "HBARUSD", "IMXUSD", "LINKUSD", "LTCUSD", "MKRUSD", "SANDUSD", "UNIUSD",
                  "USDCUSD", "XLMUSD", "XRPUSD"]
    symbolsAll = ["AAVEUSD", "ADAUSD", "ALGOUSD", "ALPHAUSD", "AMPUSD", "ANTUSD", "AUDIOUSD",
                  "AVAXUSD", "AXSUSD", "BATUSD", "BCHUSD", "BTCUSD", "CELUSD", "CHZUSD",
                  "COMPUSD", "CRVUSD", "CTSIUSD", "CVXUSD", "DAIUSD", "DYDXUSD", "ENJUSD",
                  "ETHUSD", "FETUSD", "FTMUSD", "GALAUSD", "GODSUSD", "GRTUSD", "GUSDUSD",
                  "HBARUSD", "IMXUSD", "KNCUSD", "LINKUSD", "LTCUSD", "MATICUSD", "MKRUSD", 
                  "NEXOUSD", "PAXUSD", "PERPUSD", "RADUSD", "RGTUSD", "SANDUSD", "SGBUSD",
                  "SKLUSD", "SLPUSD", "SNXUSD", "STORJUSD", "SUSHIUSD", "SXPUSD", "UMAUSD", 
                  "UNIUSD", "USDCUSD", "XLMUSD", "XRPUSD", "YFIUSD", "ZRXUSD"]
    
    print("Start")
    con = duckdb.connect(DB_PATH)
    print("Done 1")
    setup_database(con)
    print("Done 2")
    seen = set()
    for tf in TIMEFRAMES:
        for f in (DATA_POS / tf).glob("*.csv.gz"):
            print(f"Processing {f}...")
            melt_file_to_duck(con, f, seen)
    print("Done 3")
    load_riskfree(con, RISKFREE)
    print("Done 4")
    apply_holding_period_sql(con, periods=6, unit="MINUTE", bar_size=15,
                         symbols=symbolsAll, batch_rules_non_excl=300,
                         batch_rules_excl=200, chunk_months_excl=6)
    add_zero_rules_batched(con, chunk_months=12, threads=4)
    con.execute("CREATE OR REPLACE TABLE pos_all AS SELECT * FROM pos_all_extended;")
    con.execute("DROP TABLE pos_all_extended;")
    print("✓ pos_all ready (non-zero expansion + zero-only rules added).")
    print("Done 5")
    create_rule_ret_sql(con, chunk_months=12)
    print("Done 6")
    aggregate_daily_returns_polars_batched(con)
    print("Done 7")
    create_performance_table(con)
    print("Done 8")
    create_daily_returns_clean(con, symbolsAll, batch_size=5)
    print("Done 9")
    calculate_monthly_pvalues_batched(con, bootstrap_samples = 999, batch_size = 250)
    print("Done 10")
    fdr_selection_polars_monthly(con)
    print("Done 11")
    print("→ Checking p_values table...")
    print("Total rules in p_values:", con.sql("SELECT COUNT(*) FROM p_values_monthly").fetchone()[0])
    print("Min/Max p-values:", con.sql("SELECT MIN(p), MAX(p) FROM p_values_monthly").fetchone())
    rolling_portfolios_streamlined(con)
    print("Done 12")

    # — export perf_pooled —
    print("→ Exporting perf_pooled to 'perf_pooled.csv'…")
    df_perf = con.sql("SELECT * FROM perf_pooled").df()
    df_perf.to_csv(RESULTS_DIR / "perf_pooled.csv", index=False)

    # — export long‑form portfolio_ret —
    print("→ Exporting portfolio_ret to 'portfolio_ret.csv'…")
    df_port = con.sql("SELECT * FROM portfolio_ret").df()
    df_port.to_csv(RESULTS_DIR / "portfolio_ret.csv", index=False)
    con.close()
    
    print("🎉 Done with full Polars-based pipeline.")

if __name__ == "__main__":
    main()
