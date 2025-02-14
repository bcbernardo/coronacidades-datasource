VERSION=1.0

LOADER_IMAGE_TAG=impulsogov/simulacovid:$(VERSION)-loader
SERVER_IMAGE_TAG=impulsogov/simulacovid:$(VERSION)-server

PWD=$(shell pwd -P)

build-and-run-all: loader-build-run server-build-run

# LOADER
loader-remove:
	docker rm -f datasource-loader 2>/dev/null || true

# Build image
loader-build: loader-remove
	docker build \
		-f loader.dockerfile \
		-t $(LOADER_IMAGE_TAG) .

# Run for production environment
loader-run: loader-remove
	touch $(PWD)/.env
	chmod +x $(PWD)/src/loader/*.sh
	docker run -it --rm \
		--name datasource-loader \
		-v "$(PWD)/.env:/app/.env:ro" \
		-v "$(PWD)/src/loader:/app/src/:ro" \
		-v "datasource:/output" \
		$(LOADER_IMAGE_TAG)

# DEBUGGING Run for dev environment
loader-run-shell: loader-remove
	docker run --rm -it \
		--entrypoint "/bin/bash" \
		-v "$(PWD)/.env:/app/.env:ro" \
		-v "$(PWD)/src/loader:/app/src/:ro" \
		-v "datasource:/output" \
		$(LOADER_IMAGE_TAG)

# Groups
loader-build-run: loader-build loader-run
loader-shell: loader-build loader-run-shell


# SERVER
server-remove:
	docker rm -f datasource-server 2>/dev/null || true

# Build image
server-build: server-remove
	docker build \
		-f server.dockerfile \
		-t $(SERVER_IMAGE_TAG) .

# Run for production environment
server-run: server-remove
	docker run -d --restart=unless-stopped \
		--name datasource-server \
		-p 7000:7000 \
		-v "datasource:/output" \
		$(SERVER_IMAGE_TAG)

# DEBUGGING Run for dev environment
server-run-shell: server-remove
	docker run --rm -it \
		--entrypoint "/bin/bash" \
		-p 7000:7000 \
		-v "datasource:/output" \
		$(SERVER_IMAGE_TAG)

server-run-dev: server-remove
	docker run -d --restart=unless-stopped \
		--net=my-network \
		--name datasource-server \
		-p 7000:7000 \
		-v "datasource:/output" \
		$(SERVER_IMAGE_TAG)

create-network:
	docker network create -d bridge my-network

# Groups
server-dev: create-network server-build server-run-dev
server-build-run: server-build server-run
server-shell: server-build server-run-shell

# ANALYSIS VENV
loader-create-env-analysis:
	virtualenv .loader-anaylsis
	source .loader-anaylsis/bin/activate; \
			pip3 install --upgrade -r requirements-analysis.txt; \
			python -m ipykernel install --user --name=loader-anaylsis