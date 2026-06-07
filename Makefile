.PHONY: install download stage1 stage2 sweeps final report test reproduce clean

install:
	uv sync --frozen

download:
	uv run data-science download --dataset PD
	uv run data-science download --dataset CT

stage1:
	uv run data-science stage-1 --config configs/parkinsons.yaml
	uv run data-science stage-1 --config configs/covertype.yaml

sweeps:
	uv run data-science sweeps --config configs/parkinsons.yaml
	uv run data-science sweeps --config configs/covertype.yaml

stage2:
	uv run data-science stage-2 --config configs/parkinsons.yaml
	uv run data-science stage-2 --config configs/covertype.yaml

final:
	uv run data-science final --config configs/parkinsons.yaml
	uv run data-science final --config configs/covertype.yaml

report:
	cd reports && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex

test:
	uv run pytest

reproduce: install download stage1 sweeps stage2 final report

clean:
	rm -rf artifacts/scratch
	rm -f reports/*.aux reports/*.log reports/*.out reports/*.toc reports/*.fls reports/*.fdb_latexmk