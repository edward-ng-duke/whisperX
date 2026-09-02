# WhisperX deployment helpers — thin wrapper around docker-compose.yml.
#
#   make deploy            # start the GPU container detached, wait for /health
#   make deploy PROFILE=cpu
#   make build             # (re)build the image for $(PROFILE)
#   make logs / ps / down / restart / clean

PROFILE      ?= gpu
COMPOSE      := docker compose --profile $(PROFILE)
HEALTH_URL   ?= http://localhost:8000/health
HEALTH_WAIT  ?= 60

.DEFAULT_GOAL := help

.PHONY: help build deploy up down restart logs ps clean

help: ## Show this help
	@echo "WhisperX deployment targets (PROFILE=$(PROFILE)):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the image for $(PROFILE)
	$(COMPOSE) build

deploy: ## Start container detached and wait until /health is healthy
	$(COMPOSE) up -d
	@echo "Waiting up to $(HEALTH_WAIT)s for $(HEALTH_URL) ..."
	@for i in $$(seq 1 $(HEALTH_WAIT)); do \
		if curl -fsS $(HEALTH_URL) >/dev/null 2>&1; then \
			echo "✅ healthy after $${i}s"; \
			curl -fsS $(HEALTH_URL); echo; \
			$(COMPOSE) ps; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "❌ not healthy within $(HEALTH_WAIT)s — recent logs:"; \
	$(COMPOSE) logs --tail=50; \
	exit 1

up: ## Start container in the foreground (attached)
	$(COMPOSE) up

down: ## Stop and remove the container
	$(COMPOSE) down

restart: down deploy ## Recreate the container

logs: ## Follow container logs
	$(COMPOSE) logs -f

ps: ## Show container status
	$(COMPOSE) ps

clean: down ## Stop container and prune dangling state (keeps images)
	-docker container prune -f
