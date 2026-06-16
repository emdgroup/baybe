# Streamlit demos

Interactive playgrounds for exploring BayBE surrogates. The most relevant one here
is the **GP Mean Transfer Demo** (`gp_mean_transfer_demo.py`).

## Requirements

Beyond a working BayBE install, these apps need three extra packages that are **not**
part of BayBE's declared dependencies:

- `streamlit`
- `bokeh`
- `streamlit-bokeh3-events`

## Running the app (existing uv environment)

If you already have a uv environment with BayBE installed, just add the extra
packages into it and launch the app:

```bash
# from the repository root, with your uv environment active
uv pip install streamlit "bokeh>=3.3,<4" "streamlit-bokeh3-events>=0.1.4"
streamlit run streamlit/gp_mean_transfer_demo.py
```

## Running the app (fresh environment)

```bash
# from the repository root
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install streamlit "bokeh>=3.3,<4" "streamlit-bokeh3-events>=0.1.4"
streamlit run streamlit/gp_mean_transfer_demo.py
```

The app opens in your browser (default: <http://localhost:8501>).
