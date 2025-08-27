#!/bin/bash
# Start nginx
service nginx start

# Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# Keep container running
tail -f /dev/null