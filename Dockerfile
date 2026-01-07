FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src/ /app
WORKDIR /app

# Override the Lambda entrypoint and run uvicorn directly
ENTRYPOINT ["uvicorn", "server:app"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
