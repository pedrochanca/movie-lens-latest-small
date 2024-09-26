# Use the official Python image as a base.
FROM python:3.11

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container's working directory
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Set default command.
CMD ["python3"]
