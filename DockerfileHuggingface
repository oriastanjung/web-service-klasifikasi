# Use an official Python runtime as a parent image
FROM python:3.10.14-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get clean

# Install Uvicorn
RUN pip3 install uvicorn

# Make port 5000 available to the world outside this container
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
