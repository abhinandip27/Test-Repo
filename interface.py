import time
import pandas as pd
import plotly.express as px
import streamlit as st 
import signal_buffer
import pleth_signalprocessing

signalbuffer = signal_buffer.SignalBuffer(signal_buffer.SIGNALSOURCE_FILE, 2550, {"Filename": "Datasets/Set-1/data_ppg.csv"})
signalproc = pleth_signalprocessing.PlethSignalProcessor(signalbuffer)

st.set_page_config(page_title="Remote Monitoring - PPG Signals", page_icon="✅", layout="wide")
col1, col2 = st.columns([0.8, 0.2])
col1.title("Plethysmograph - Dashboard and feature monitor")
col2.image("images/iistlogo.jpeg")
col2.markdown("## IIST, Trivandrum")
col2.markdown("### KSCSTE Project")

# creating a single-element container
placeholder = st.empty()

while True:
	with placeholder.container():
		st.markdown("### Remote monitor (PPG/MPG) signal")
		df = pd.DataFrame(columns = ["Time", "PPG"])
		df["Time"] = signalbuffer.buffer_time
		df["PPG"] = signalbuffer.buffer_signal
		fig = px.line(data_frame=df, y="PPG", x="Time")
		st.write(fig)
		
		ftr = signalproc.get_feature()
		ftr_series = signalproc.get_feature_series()

		hr_series = ftr_series["HR"]
		t_hr = []
		hr_hr = []
		while not hr_series.empty():
			tt, hrt = hr_series.get()
			t_hr.append(tt)
			hr_hr.append(hrt)

		col3, col4, col5, col6 = st.columns([0.5, 0.5, 0.5, 0.5])
		col3.markdown("### Estimated Heart Rate: %.1f @ %.1f" % (ftr["HR"][1], ftr["HR"][0]))

		col4.markdown("### Average Systolic Amplitude: %.1f @ %.1f" % (ftr["SysA"][1], ftr["SysA"][0]))

		col5.markdown("### Heart Rate Variability: %.3f @ %.1f" % (ftr["HRV"][1], ftr["HRV"][0]))

		col5.markdown("### Augmentation Index: %.1f @ %.1f" % (ftr["AI"][1], ftr["AI"][0]))

		signalbuffer.update_buffer()
		time.sleep(0.1)