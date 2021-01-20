import pandas as pd
import numpy as np
import datetime as dt
import yaml
from math import ceil

from endpoints import (
    get_health,
    get_places_id,
    get_states_cases,
    get_states_rt,
    get_health_region_farolcovid_main,
)
from endpoints.get_cities_farolcovid_main import (
    get_situation_indicators,
    get_control_indicators,
    get_capacity_indicators,
    get_trust_indicators,
    get_overall_alert,
)
from endpoints.get_cities_rt import get_rt
from endpoints.helpers import allow_local
from utils import download_from_drive


def _daterange(start_date, end_date):
    """Generator of dates between a start and an end dates."""
    # CREDIT: https://stackoverflow.com/a/1060330
    for n in range(int((end_date - start_date).days)):
        yield start_date + pd.Timedelta(n, unit="D")


def _get_population_by_state(config, country="br"):
    """Get population by state."""

    # adapted from endpoints.get_health

    # download population (by city) spreadsheet
    df = download_from_drive(
        config[country]["drive_paths"]["cities_population"]
    )
    
    # Fix for default places ids - before "health_system_region"
    places_ids = get_places_id.now(config).assign(
        city_id=lambda df: df["city_id"].astype(int)
    )
    df = df.drop(["city_name", "state_name"], axis=1).merge(
        places_ids, on=["city_id", "state_id"]
    )

    # Fix date types
    time_cols = [c for c in df.columns if "last_updated" in c]
    df[time_cols] = df[time_cols].apply(pd.to_datetime)

    # adapted from endpoints.get_states_farolcovid_main.now()
    # sum all cities in state
    df = (
        df
        .groupby(
            [
                "country_iso",
                "country_name",
                "state_num_id",
                "state_id",
                "state_name",
            ]
        )
        .agg({"population": "sum"})
        .reset_index()
        .sort_values("state_num_id")
        .set_index("state_num_id")
    )

    return df


@allow_local
def now(config):
    """Data & states' indicators for FarolCovid app, keeping past records."""

    # adapted from endpoints.get_states_farolcovid_main.now()

    # NOTE: this version do not include health resources and capacity data,
    # as these indicators come from a source that do not track historical data

    df = pd.DataFrame()

    # get full record of case counts for brazilian states
    cases_latest = get_states_cases.now(config)
    cases_latest["last_updated"] = pd.to_datetime(cases_latest["last_updated"])

    # population by state (fixed)
    population_df = _get_population_by_state(config)

    # iterate over each date since the first case (Feb. 25th, 2020)
    for date in _daterange(pd.Timestamp(2020, 2, 25), pd.Timestamp.today()):
        cases_until_then = (
            cases_latest.loc[cases_latest["last_updated"] <= date, :]
        )

        df_partial = get_situation_indicators(
            population_df,
            data=cases_until_then,
            place_id="state_num_id",
            rules=config["br"]["farolcovid"]["rules"],
            classify="situation_classification",
        )

        try:
            df_partial = get_control_indicators(
                df_partial,
                data=get_rt(cases_until_then, "state_num_id", config),
                place_id="state_num_id",
                rules=config["br"]["farolcovid"]["rules"],
                classify="control_classification",
            )
        except ValueError:
            # raised when at least one of the states have no record in the last
            # ten days (specially in the beginning of the epidemics)
            pass

        df_partial = get_trust_indicators(
            df_partial,
            data=cases_until_then,
            place_id="state_num_id",
            rules=config["br"]["farolcovid"]["rules"],
            classify="trust_classification",
        )

        # alert classification
        cols = [col for col in df_partial.columns if "classification" in col]
        df_partial["overall_alert"] = df_partial.apply(
            lambda row: get_overall_alert(row[cols]), axis=1
        )

        # add to historical data
        df = pd.concat([df, df_partial], ignore_index=True)
    
    return df_partial




TESTS = {
    "doesnt have 27 states": lambda df: len(df["state_id"].unique()) == 27,
    "df is not pd.DataFrame": lambda df: isinstance(df, pd.DataFrame),
    "overall alert > 3": lambda df: all(
        df[~df["overall_alert"].isnull()]["overall_alert"] <= 3
    ),
    "doesnt have both rt classified and growth": lambda df: df[
        "control_classification"
    ].count()
    == df["rt_most_likely_growth"].count(),
    "rt 10 days maximum and minimum values": lambda df: all(
        df[
            ~(
                (df["rt_low_95"] < df["rt_most_likely"])
                & (df["rt_most_likely"] < df["rt_high_95"])
            )
        ]["rt_most_likely"].isnull()
    ),
}
