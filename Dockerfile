# Use a lightweight Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create the required folder structure inside the container
RUN mkdir -p /app/processed /app/uploads /app/logs /app/worker/data/in /app/worker/data/results /app/worker/data/summary

# Expose any necessary ports (if applicable)
# EXPOSE 5000

# Specify the default command
CMD ["python", "-u", "main.py"]
