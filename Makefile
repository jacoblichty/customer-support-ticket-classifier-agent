# Docker Compose commands for different environments

# Development
dev:
	docker-compose --profile dev up --build ticket-classifier-dev

# Production
prod:
	docker-compose up --build -d ticket-classifier

# Production with all services
prod-full:
	docker-compose --profile production up --build -d

# Testing
test:
	docker build --target testing -t ticket-classifier:test .
	docker run --rm ticket-classifier:test

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Logs
logs:
	docker-compose logs -f ticket-classifier

# Shell access
shell:
	docker-compose exec ticket-classifier /bin/bash

# Health check
health:
	curl -f http://localhost:8000/health

# Load test
load-test:
	docker run --rm -it --network="host" \
		-v $(PWD)/tests:/tests \
		python:3.11-slim \
		bash -c "pip install httpx && python /tests/load_test.py"

.PHONY: dev prod prod-full test clean logs shell health load-test