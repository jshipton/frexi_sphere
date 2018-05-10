test:
	@echo "    Running all tests"
	#@py.test tests $(PYTEST_ARGS)
	@python3 -m pytest tests $(PYTEST_ARGS)
