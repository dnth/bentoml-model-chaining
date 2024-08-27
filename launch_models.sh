#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to start a service
start_service() {
    local service=$1
    local port=$2
    cd $service/
    bentoml serve . -p $port &
    cd ..
    echo -e "${GREEN}Started $service service on port $port${NC}"
}

# Function to prompt user for service selection
select_services() {
    echo -e "${BLUE}Select services to start (comma-separated numbers, or 'all'):${NC}"
    echo -e "${YELLOW}1) RAM (port 3000)"
    echo "2) YOLO (port 3001)"
    echo "3) BLIP2 (port 3002)"
    echo "4) OWLv2 (port 3003)"
    echo "5) Phi 3.5 (port 3004)"
    echo -e "6) All services${NC}"
    read -p "$(echo -e ${BLUE}Enter your choice: ${NC})" choice
}

# Prompt user for service selection
select_services

# Start selected services
if [[ $choice == "6" || $choice == "all" ]]; then
    echo -e "${YELLOW}Starting all services...${NC}"
    start_service "ram" 3000
    start_service "yolo" 3001
    start_service "blip2" 3002
    start_service "owlv2" 3003
    start_service "phi3.5-vision" 3004
else
    IFS=',' read -ra SERVICES <<< "$choice"
    for num in "${SERVICES[@]}"; do
        case $num in
            1) start_service "ram" 3000 ;;
            2) start_service "yolo" 3001 ;;
            3) start_service "blip2" 3002 ;;
            4) start_service "owlv2" 3003 ;;
            5) start_service "phi3.5-vision" 3004 ;;
            *) echo -e "${RED}Invalid choice: $num${NC}" ;;
        esac
    done
fi

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Launch Streamlit app
echo -e "${GREEN}Launching Streamlit app...${NC}"
streamlit run demo/app.py