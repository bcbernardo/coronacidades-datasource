import pandas as pd
import numpy as np
import datetime as dt
import yaml

from endpoints import (
    get_cases,
    get_cities_rt,
    get_health_region_rt,
    get_states_rt,
    get_health,
)
from endpoints.helpers import allow_local
from endpoints.scripts.simulator import run_simulation
from tqdm import tqdm

tqdm.pandas()


def _get_levels(df, rules):
    return pd.cut(
        df[rules["column_name"]],
        bins=rules["cuts"],
        labels=rules["categories"],
        right=False,
        include_lowest=True,
    )


# SITUATION: New cases
def get_situation_indicators(df, data, place_id, rules, classify, growth=None):

    data["last_updated"] = pd.to_datetime(data["last_updated"])
    data = data.loc[data.groupby("city_id")["last_updated"].idxmax()].set_index(
        "city_id"
    )

    df["last_updated_cases"] = data["last_updated"]
    # TODO: mudar para new_cases_1m_mavg de data!
    df["new_cases_1m_mavg"] = (
        data["daily_cases"] / data["estimated_population_2019"]
    ) * 10 ** 6

    df[classify] = _get_levels(df, rules[classify])
    # TODO: add if growth == 1, +1 level!

    return df


# CONTROL: - (no testing data!)
def get_control_indicators(df, data, place_id, rules, classify, growth=None):

    data = data.assign(last_updated=lambda df: pd.to_datetime(df["last_updated"]))
    data = data.loc[data.groupby(place_id)["last_updated"].idxmax()]

    # Min-max do Rt de 14 dias (max data de taxa de notificacao)
    df[
        ["last_updated_rt", "rt_low_95", "rt_high_95", "rt_most_likely"]
    ] = data.sort_values(place_id).set_index(place_id)[
        ["last_updated", "Rt_low_95", "Rt_high_95", "Rt_most_likely"]
    ]

    # Classificação: melhor estimativa do Rt de 10 dias (rt_most_likely)
    df[classify] = _get_levels(df, rules[classify])

    # TODO: trend on rt table!
    if growth:
        df[growth] = _get_levels(df, rules[growth])

    return df


# CAPACITY
def _calculate_recovered(df, params):

    confirmed_adjusted = int(df[["confirmed_cases"]].sum() / df["notification_rate"])

    if confirmed_adjusted == 0:  # dont have any cases yet
        params["population_params"]["R"] = 0
        return params

    params["population_params"]["R"] = (
        confirmed_adjusted
        - params["population_params"]["I"]
        - params["population_params"]["D"]
    )

    if params["population_params"]["R"] < 0:
        params["population_params"]["R"] = (
            confirmed_adjusted - params["population_params"]["D"]
        )

    return params


def _prepare_simulation(row, place_id, config, rt_upper=None):

    params = {
        "population_params": {
            "N": row["population"],
            "I": [row["active_cases"] if not np.isnan(row["active_cases"]) else 1][0],
            "D": [row["deaths"] if not np.isnan(row["deaths"]) else 0][0],
        },
        "n_beds": row["number_beds"]
        * config["br"]["simulacovid"]["resources_available_proportion"],
        "n_icu_beds": row["number_icu_beds"],
        "R0": {"best": row["rt_low_95"], "worst": row["rt_high_95"]},
    }

    # TODO: checar esses casos no calculo da subnotificacao!
    if row["notification_rate"] != row["notification_rate"]:
        return np.nan, np.nan

    if row["notification_rate"] == 0:
        return np.nan, np.nan

    # Seleciona rt de 1 nivel acima caso não tenha
    if row["rt_low_95"] != row["rt_low_95"]:

        if place_id == "city_id":
            rt = rt_upper.query(f"health_region_id == {row['health_region_id']}")
        elif place_id == "health_region_id":
            rt = rt_upper.query(f"state_num_id == {row['state_num_id']}")
        else:
            return np.nan, np.nan

        if len(rt) > 0:
            rt = rt.assign(
                last_updated=lambda df: pd.to_datetime(df["last_updated"])
            ).query("last_updated == last_updated.max()")
            params["R0"] = {"best": rt["Rt_low_95"], "worst": rt["Rt_high_95"]}
        else:
            return np.nan, np.nan

    params = _calculate_recovered(row, params)
    _, dday_icu_beds = run_simulation(params, config)

    return dday_icu_beds["best"], dday_icu_beds["worst"]


def get_capacity_indicators(df, place_id, config, rules, classify):

    if place_id == "city_id":
        rt_upper = get_health_region_rt.now(config)
    elif place_id == "health_region_id":
        rt_upper = get_states_rt.now(config)
    else:
        rt_upper = None

    df["dday_icu_beds_best"], df["dday_icu_beds_worst"] = zip(
        *df.progress_apply(
            lambda row: _prepare_simulation(row, place_id, config, rt_upper=rt_upper),
            axis=1,
        )
    )

    df["dday_icu_beds_best"] = df["dday_icu_beds_best"].replace(-1, 91)
    df["dday_icu_beds_worst"] = df["dday_icu_beds_worst"].replace(-1, 91)

    # Classificação: numero de dias para acabar a capacidade
    df[classify] = _get_levels(df, rules[classify])

    return df


# TRUST
# TODO: add here after update on cases df
def get_trust_indicators(df, data, place_id, rules, classify, growth=None):

    data["last_updated"] = pd.to_datetime(data["last_updated"])
    df["subnotification_rate"] = 1 - df["notification_rate"]

    # Última data com notificação: 14 dias atrás
    df[["last_updated_subnotification", "notification_rate"]] = (
        data.dropna().groupby("city_id")[["last_updated", "notification_rate"]].last()
    )

    # Classificação: percentual de subnotificação
    df[classify] = _get_levels(df, rules[classify])

    return df


def get_overall_alert(indicators):
    if indicators.notnull().all():
        return int(max(indicators))
    else:
        return np.nan


@allow_local
def now(config):

    # Get last cases data
    cases = (
        get_cases.now(config, "br")
        .dropna(subset=["active_cases"])
        .assign(last_updated=lambda df: pd.to_datetime(df["last_updated"]))
    )
    cases = cases.loc[cases.groupby("city_id")["last_updated"].idxmax()].drop(
        config["br"]["cases"]["drop"] + ["state_num_id", "health_region_id"], 1
    )

    # Merge resource data
    df = get_health.now(config, "br")[
        config["br"]["simulacovid"]["columns"]["cnes"]
    ].merge(cases, on="city_id", how="left")

    df = (
        df[config["br"]["farolcovid"]["simulacovid"]["columns"]]
        .sort_values("city_id")
        .set_index("city_id")
        .assign(confirmed_cases=lambda x: x["confirmed_cases"].fillna(0))
        .assign(deaths=lambda x: x["deaths"].fillna(0))
    )

    # TODO: mudar indicadores de situacao + add trust (notification_rate)!
    df = get_situation_indicators(
        df,
        data=get_cases.now(config),
        place_id="city_id",
        rules=config["br"]["farolcovid"]["rules"],
        classify="situation_classification",
        growth=None,  # -> "situation_growth" depois do update na tabela de casos
    )

    df = get_control_indicators(
        df,
        data=get_cities_rt.now(config),
        place_id="city_id",
        rules=config["br"]["farolcovid"]["rules"],
        classify="control_classification",
        growth=None,  # "control_growth",
    )

    df = get_capacity_indicators(
        df,
        place_id="city_id",
        config=config,
        rules=config["br"]["farolcovid"]["rules"],
        classify="capacity_classification",
    )

    df = get_trust_indicators(
        df,
        data=get_cases.now(config),
        place_id="city_id",
        rules=config["br"]["farolcovid"]["rules"],
        classify="trust_classification",
        growth=None,  # "trust_growth",
    )

    cols = [col for col in df.columns if "classification" in col]
    df["overall_alert"] = df.apply(
        lambda row: get_overall_alert(row[cols]), axis=1
    )  # .replace(config["br"]["farolcovid"]["categories"])

    return df.reset_index()


TESTS = {
    "doesnt have 5570 cities": lambda df: len(df["city_id"].unique()) == 5570,
    "doesnt have 27 states": lambda df: len(df["state_num_id"].unique()) == 27,
    "df is not pd.DataFrame": lambda df: isinstance(df, pd.DataFrame),
    # "city doesnt have both rt classified and growth": lambda df: df[
    #     "control_classification"
    # ].count()
    # == df["control_growth"].count(),
    "dday worst greater than best": lambda df: len(
        df[df["dday_icu_beds_best"] < df["dday_icu_beds_worst"]]
    )
    == 0,
    "rt 10 days maximum and minimum values": lambda df: all(
        df[
            ~(
                (df["rt_low_95"] < df["rt_most_likely"])
                & (df["rt_most_likely"] < df["rt_high_95"])
            )
        ]["rt_most_likely"].isnull()
    ),
    "city with all classifications got null alert": lambda df: all(
        df[df["overall_alert"].isnull()][
            [
                "control_classification",
                "situation_classification",
                "capacity_classification",
                "trust_classification",
            ]
        ]
        .isnull()
        .apply(lambda x: any(x), axis=1)
        == True
    ),
    "city without classification got an alert": lambda df: all(
        df[
            df[
                [
                    "capacity_classification",
                    "control_classification",
                    "situation_classification",
                    "trust_classification",
                ]
            ]
            .isnull()
            .any(axis=1)
        ]["overall_alert"].isnull()
        == True
    ),
}
