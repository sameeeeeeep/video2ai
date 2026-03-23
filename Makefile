.PHONY: install dev build publish clean

install:
	pip install -e ".[vision]"

dev:
	pip install -e ".[vision]" && brew install ffmpeg

build:
	pip install build && python -m build

publish: build
	pip install twine && twine upload dist/*

binary:
	pip install pyinstaller
	pyinstaller --name video2ai --onefile \
		--hidden-import video2ai --hidden-import video2ai.cli \
		--hidden-import video2ai.web --hidden-import video2ai.frames \
		--hidden-import video2ai.probe --hidden-import video2ai.transcribe \
		--hidden-import video2ai.vision --hidden-import video2ai.clip_match \
		--hidden-import video2ai.contact_sheet --hidden-import video2ai.embed \
		--hidden-import video2ai.llm --hidden-import video2ai.output \
		--collect-data whisper \
		video2ai/cli.py

clean:
	rm -rf build/ dist/ *.egg-info video2ai.egg-info __pycache__

release:
	@echo "Usage: make release VERSION=0.2.0"
	@test -n "$(VERSION)" || (echo "VERSION is required" && exit 1)
	sed -i '' 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml
	sed -i '' "s/__version__ = \".*\"/__version__ = \"$(VERSION)\"/" video2ai/__init__.py
	git add pyproject.toml video2ai/__init__.py
	git commit -m "release: v$(VERSION)"
	git tag v$(VERSION)
	git push origin main --tags
