.PHONY: build run stop

# Build the Docker image
build:
	docker build -t ta-sage .

# Run the Docker container
run:
	docker run --gpus all -d --name ta-sage-container-3 -v "$(CURDIR)":/app -v /mnt/nfs/sato:/mnt/ ta-sage tail -f /dev/null

# Stop and remove the Docker container
stop:
	docker stop ta-sage-container
	docker rm ta-sage-container
