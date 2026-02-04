"""#!/usr/bin/env python3
import io
import os
import traceback

import streamlit as st

from generate_solar import synthesize

st.set_page_config(page_title="Solar Generator", layout="centered")

st.title("Solar Power Dataset Generator")

days = st.number_input("Days to simulate", min_value=1, max_value=365, value=7)
samples_per_day = st.selectbox("Samples per day", options=[24, 48, 96], index=0)
seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

col1, col2 = st.columns(2)
with col1:
    generate = st.button("Generate dataset")
with col2:
    save_file = st.checkbox("Save to dataset/solar_dataset.csv", value=True)

if generate:
    with st.spinner("Generating dataset..."):
        try:
            # synthesize may be somewhat expensive; cache results for identical inputs
            df = synthesize(days=days, samples_per_day=samples_per_day, seed=int(seed))
            st.success(f"Generated {len(df)} rows")
            st.dataframe(df.head(200))

            st.subheader("Power (kW) over time")
            st.line_chart(df.set_index("timestamp_utc")["power_kw"])

            # CSV download
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            csv_data = csv_buf.getvalue().encode("utf-8")
            st.download_button("Download CSV", data=csv_data, file_name="solar_dataset.csv", mime="text/csv")

            if save_file:
                out_dir = "dataset"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "solar_dataset.csv")
                df.to_csv(out_path, index=False)
                st.info(f"Saved dataset to {out_path}")
        except Exception as e:
            st.error(f"Error while generating dataset: {e}")
            traceback.print_exc()
"""