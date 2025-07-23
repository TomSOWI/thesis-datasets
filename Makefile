#.PHONY: job

job:
	squeue -u nimtsspi

quota:
	show-quota

push:
	@read -p "Commit message: " msg; \
	git add . && git commit -m "$$msg" && git push
