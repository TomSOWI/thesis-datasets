#.PHONY: job

#job:
#	squeue -u nimtsspi

job:
	squeue -u u11191

quota:
	show-quota

push:
	@read -p "Commit message: " msg; \
	git add . && git commit -m "$$msg" && git push

faiss:
	eval "$(micromamba shell hook --shell bash)" && micromamba activate faiss

