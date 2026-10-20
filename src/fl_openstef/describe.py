"""
What the GUI says about the OpenSTEF run. No imports, so the GUI (CityLearn venv) can read it without OpenSTEF.

Two levels on purpose: the plain text is for someone who knows neither AnyLog nor forecasting;
`details` is for someone who has run forecasting in a utility for years.
"""

TASK = {
    "name": "openstef",
    "title": "OpenSTEF forecast",
    "headline": "Forecast each community's electricity use, from data that never leaves its node",
    "summary": ("A grid operator has to know tomorrow's electricity use today, to plan the grid and buy energy ahead. "
                "OpenSTEF, LF Energy's open-source forecasting tool, learns each community's pattern from its history "
                "in AnyLog and predicts the next day and a half. The forecast is then checked against what really happened."),
    "local_step": ("For each week, AnyLog adds up every building's hourly use on the community's own node and returns "
                   "one total per hour. OpenSTEF learns from all the hours before the week's last day and a half, "
                   "then forecasts that day and a half without seeing it."),
    "combine_step": "Each community gets its own model; nothing is combined.",
    "travels": ("one total per hour per community. The individual building readings stay on the node, and so does "
                "who used what."),
    "metric": "mae_kwh",
    "metric_label": "Average forecast error (kWh per hour, whole community)",
    "lower_is_better": True,
    "series": {"openstef": "OpenSTEF forecast", "baseline": "Simple guess: same hour last week"},
    "settings": [],
    "glossary": [
        ["Forecast", "OpenSTEF's best estimate for each hour (the middle of its range)."],
        ["Range", "Where OpenSTEF expects the real value to land 8 times out of 10. A narrow range means it is sure."],
        ["Simple guess", "Assume each hour repeats the same hour a week earlier. Any forecast worth using should beat it."],
        ["Error", "How far the forecast was from what really happened, on average, in kWh per hour."],
    ],
    "details": [
        ["Model", "OpenSTEF 3.4 train_model_pipeline_core / create_forecast_pipeline_core, {model} regressor, "
                  "quantiles {quantiles}, one prediction job per community."],
        ["Training data", "All hourly history before each test window ({train_days}); windows with less than "
                          "{min_train_days} days are skipped (OpenSTEF's lag features reach back 14 days)."],
        ["Inputs", "Load = SUM(net_electricity_consumption) over the community's buildings; temperature, humidity, "
                   "direct and diffuse radiation and price averaged. Hours where not every building reported are dropped."],
        ["Test window", "The last ~20% of each week (about 34 hours), held back from training: a day-ahead horizon."],
        ["Weather", "The window's measured weather stands in for a weather forecast, so the error is a little "
                    "optimistic compared with operation."],
        ["Scores", "MAE and RMSE in kWh per hour; P10-P90 coverage is the share of hours inside the 80% band "
                   "(well calibrated ≈ 0.80). Baseline: same hour one week earlier (seasonal naive, 168 h)."],
        ["Data access", "Every query goes through AnyLog: the UNS gives each community's dbms, tables and per-building "
                        "WHERE; one network query per window spans the train and test tables (include=, @table_name) "
                        "and aggregates at the node with increments(hour, 1, timestamp)."],
    ],
}