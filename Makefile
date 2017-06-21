lint:
	@echo "    Linting frexi codebase"
	@flake8 frexi_sphere --exclude rexi_error.py
	@echo "    Linting frexi tests"
	@flake8 tests

test:
	@echo "    Running all tests"
	@py.test tests $(PYTEST_ARGS)
