# Use an official Python runtime as base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list first (helps with caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (Flask default is 5000)
EXPOSE 5000

# Command to run Flask API
CMD ["python", "app.py"]
