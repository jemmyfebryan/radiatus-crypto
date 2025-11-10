# Save this file as: ./Makefile

# Name of the app as defined in ecosystem.json
APP_NAME = "signal-backend"
# Path to the ecosystem file
ECO_FILE = "ecosystem.json"

.PHONY: all install start stop restart logs status delete

all: start

# Install python dependencies
install:
	@echo "--- Installing Python dependencies from requirements.txt ---"
	pip install -r requirements.txt

# Start the application in production mode
start:
	@echo "--- Starting application with PM2 ---"
	pm2 start $(ECO_FILE) --env production

# Stop the application
stop:
	@echo "--- Stopping application ---"
	pm2 stop $(ECO_FILE)

# Restart the application
restart:
	@echo "--- Restarting application ---"
	pm2 restart $(ECO_FILE) --env production

# Show logs for the application
logs:
	@echo "--- Tailing logs (Ctrl+C to exit) ---"
	pm2 logs $(APP_NAME) --raw

# Show the status of all PM2 processes
status:
	@echo "--- PM2 process status ---"
	pm2 list

# Stop and delete the application from PM2's list
delete:
	@echo "--- Stopping and deleting application from PM2 ---"
	pm2 delete $(ECO_FILE)