
# pull the docker image
FROM python:3.9.20-slim

# set the working directory
WORKDIR /dsts

# copy the contents to the working directory
COPY . /dsts

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# run the application
CMD [ "python", "part-b.py" ]