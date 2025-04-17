
class GeoPsychMarketForecaster:
    def __init__(self):
        self.forecasts = {"Asia": "neutral", "Europe": "confident", "America": "nervous"}

    def forecast_by_region(self, region):
        region = region.lower()
        if region == "asia":
            return self.forecasts["Asia"]
        elif region == "europe":
            return self.forecasts["Europe"]
        elif region == "america":
            return self.forecasts["America"]
        return "unknown"

    def update_forecast(self, region, new_forecast):
        if region.lower() in self.forecasts:
            self.forecasts[region.lower()] = new_forecast
            return f"Forecast for {region} updated to {new_forecast}."
        return "Region not found."

    def get_forecasts(self):
        return self.forecasts
