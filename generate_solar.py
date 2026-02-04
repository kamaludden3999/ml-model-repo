#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def synthesize(days: int = 7, samples_per_day: int = 24, seed: int = 42):
    np.random.seed(seed)
    total = days * samples_per_day
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [start + timedelta(hours=i) for i in range(total)]

    # Simulate clear-sky irradiance as a daily sinusoid (W/m^2)
    hours = np.array([t.hour + t.minute/60.0 for t in timestamps])
    day_fraction = (np.arange(total) % samples_per_day) / float(samples_per_day)
    # Peak irradiance around midday ~1000 W/m^2
    irradiance = np.maximum(0, 1000 * np.sin(np.pi * day_fraction))
    # Add some weather noise and occasional cloud dips
    irradiance *= np.clip(1 + 0.2 * np.random.randn(total), 0.3, 1.4)
    cloud_mask = (np.random.rand(total) < 0.05)
    irradiance[cloud_mask] *= np.random.uniform(0.0, 0.4, cloud_mask.sum())

    # Temperatures: base diurnal cycle around 20C with some noise
    temperature = 20 + 8 * np.sin(2 * np.pi * (hours - 6) / 24.0) + 2 * np.random.randn(total)

    # Simple PV formula: power = area * efficiency * irradiance * temperature_factor
    panel_area = 10.0  # m^2 (example)
    efficiency = 0.18   # 18% module efficiency
    temp_coeff = -0.004  # relative change per degree above 25C

    temp_factor = 1.0 + temp_coeff * (temperature - 25.0)
    power_w = panel_area * efficiency * irradiance * temp_factor
    power_w = np.clip(power_w, 0, None)
    # Convert to kW
    power_kw = power_w / 1000.0

    df = pd.DataFrame({
        'timestamp_utc': [t.isoformat() for t in timestamps],
        'irradiance_w_m2': np.round(irradiance, 2),
        'temperature_c': np.round(temperature, 2),
        'power_kw': np.round(power_kw, 4)
    })
    return df

def main():
    parser = argparse.ArgumentParser(description='Synthesize solar power generation time series and save CSV')
    parser.add_argument('--days', type=int, default=7, help='Number of days to simulate')
    parser.add_argument('--samples-per-day', type=int, default=24, help='Samples per day (e.g., 24 for hourly)')
    parser.add_argument('--output', type=str, default='dataset/solar_dataset.csv', help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    df = synthesize(days=args.days, samples_per_day=args.samples_per_day, seed=args.seed)
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Wrote {len(df)} rows to {args.output}')

if __name__ == '__main__':
    main()