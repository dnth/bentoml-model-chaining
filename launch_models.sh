#!/bin/bash

# Start YOLO service
cd yolo/
bentoml serve . -p 3001 &
cd ..

# Start BLIP2 service
cd blip2/
bentoml serve . -p 3002 &
cd ..

# Start RAM service
cd ram/
bentoml serve . -p 3000 &
cd ..

# Start OWLv2 service
cd owlv2/
bentoml serve . -p 3003 &
cd ..

# Wait for services to start
sleep 10

# Launch Streamlit app
streamlit run demo/app.py