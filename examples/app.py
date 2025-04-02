import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON files
strs = os.listdir('./')
df = []
cwd = os.getcwd()
for p in strs:
    if p.endswith(".json"):
        data = json.load(open(f"{cwd}/{p}", "r"))
        df.extend(data)
df = pd.DataFrame(df)

# Add learning rate and optimizer info
lr = '1e-3'
optimizer = 'adam'
df['lr'] = lr
df['optimizer'] = optimizer

# Preview data
st.write("Data Preview:")
st.dataframe(df.head())

# Module selection (multi-select)
module_list = df['module'].unique()
selected_modules = st.multiselect("Select module(s)", module_list, default=module_list[:1])

# X-axis selection
x_axis = st.selectbox("Select x-axis", ['width', 't'])

# Filter based on selected modules
filtered_df = df[df['module'].isin(selected_modules)]

# Let user choose a value for the other axis
if x_axis == 'width':
    t_values = filtered_df['t'].unique()
    selected_t = st.selectbox("Filter by t", sorted(t_values))
    filtered_df = filtered_df[filtered_df['t'] == selected_t]
else:
    width_values = filtered_df['width'].unique()
    selected_width = st.selectbox("Filter by width", sorted(width_values))
    filtered_df = filtered_df[filtered_df['width'] == selected_width]

# Plotting
fig, ax = plt.subplots()
for module in selected_modules:
    module_df = filtered_df[filtered_df['module'] == module]
    ax.plot(module_df[x_axis], module_df['l1'], marker='o', label=module)

ax.set_xlabel(x_axis)
ax.set_ylabel("l1")
ax.set_title(f"{x_axis} vs l1")
ax.legend()
st.pyplot(fig)
