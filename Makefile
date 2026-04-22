.PHONY: ci lint

ci: lint

lint:
	@echo "Running flake8..."
	flake8 .
