.PHONY: install test train serve lint clean

install:
	poetry install --with dev

test:
	poetry run pytest tests/ -v

train:
	poetry run dojo train --config configs/training/alphazero_connect_four.yaml

serve:
	@test -n "$(CHECKPOINT)" || (echo "Usage: make serve CHECKPOINT=path/to/best.pt" && exit 1)
	poetry run dojo serve --checkpoint "$(CHECKPOINT)"

lint:
	poetry run python -m py_compile src/dojo/core/types.py

clean:
	rm -rf .venv __pycache__ dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
