.PHONY: test cov clean

# 默认目标：全量测试 + 覆盖率报告
test:
	python -m pytest tests/ -v

# 仅运行四个验收用例
test-required:
	python -m pytest tests/test_required.py -v --no-cov

# 仅运行 unit 标记（快速）
test-unit:
	python -m pytest tests/ -m unit -v --no-cov

# 生成 HTML 覆盖率报告（浏览器查看）
cov:
	python -m pytest tests/ --cov=maze_env \
		--cov-report=html:reports/coverage \
		--cov-report=term-missing \
		--cov-fail-under=90

# 清理所有生成产物（含训练产物）
clean:
	rm -rf reports/ .coverage .pytest_cache \
		__pycache__ src/__pycache__ tests/__pycache__ \
		maze_env/__pycache__ *.egg-info dist build
