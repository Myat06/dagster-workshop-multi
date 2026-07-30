import pandas as pd
from dagster import Definitions, ScheduleDefinition, asset, define_asset_job

import db
import source


@asset
def raw_exchange_rates() -> pd.DataFrame:
    payload = source.fetch_latest_rates(base="USD")
    rows = [
        {"base_currency": "USD", "quote_currency": currency, "rate": rate}
        for currency, rate in payload["rates"].items()
    ]
    return pd.DataFrame(rows)


@asset
def exchange_rates_table(raw_exchange_rates: pd.DataFrame) -> int:
    return db.load_table(raw_exchange_rates, "exchange_rates")


@asset
def orders_in_eur(exchange_rates_table: int) -> pd.DataFrame:
    engine = db.get_engine()
    orders = pd.read_sql("SELECT * FROM orders", engine)
    products = pd.read_sql("SELECT * FROM products", engine)
    rates = pd.read_sql("SELECT * FROM exchange_rates", engine)

    merged = orders.merge(products, on="product_id")
    merged["order_value_usd"] = merged["quantity"] * merged["price"]

    eur_rate = rates.loc[rates["quote_currency"] == "EUR", "rate"].iloc[0]
    merged["order_value_eur"] = merged["order_value_usd"] * eur_rate

    return merged[
        ["order_id", "customer_id", "product_id", "order_value_usd", "order_value_eur"]
    ]


refresh_fx_job = define_asset_job(name="refresh_fx_job")

refresh_fx_daily = ScheduleDefinition(
    name="refresh_fx_daily",
    job=refresh_fx_job,
    cron_schedule="0 6 * * *",
)

defs = Definitions(
    assets=[raw_exchange_rates, exchange_rates_table, orders_in_eur],
    jobs=[refresh_fx_job],
    schedules=[refresh_fx_daily],
)
