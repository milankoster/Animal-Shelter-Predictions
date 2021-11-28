# README

Animal Shelter Predictions is a challenge aimed at prediction how long it will take for cats and dogs to be adopted using classification. 

The project is build up of 2 delivery documents and 4 main notebooks.

**Delivery Documents**
1. `Shelter Animals - Technical Delivery`
	* Describes the project, the steps which were taken and the technologies which were used.
	* Functions as a GitHub README.
	* Aimed at recruiters, teachers and technical readers.
2. `Animal Shelter Infograph`
	* Described what the project was about in a non-technical sense.
	* Aimed at Animal Shelters and/or governments who don't know much about machine learning and programming.


**Notenooks**
The notebooks are created and should be read in the below order. Some parts are overlapping. You will find parts of Data Preparation in the EDA notebook and some visualisations in the Modelling notebook.

3. `Shelter Animals - Proposal`
4. `Shelter Animals - Data Preparation`
5. `Shelter Animals - Exploratory Data Analysis`
6. `Shelter Animals - Modelling`


**Deployment**
I deployed my model using a Flask API that runs as a local webserver. To run the API:
- Activate the virtual environment flask-app-venv
- Install the requirements using the command `pip install -r requirements.txt`
- Activate the file `flask-app-venv/Scripts/activate.bat`
- Send a POST request to `127.0.0.1:5000/predict/` with correct JSON format

An example of correct JSON is as follows:
```json
{
	"Animal Type": "Cat",
	"Breed": "Domestic Medium Hair",
	"Gender": "Female",
	"Intake Type": "Stray",
	"Intake Condition": "Normal",
	"Castration Intake": false,
	"Castration Current": true,
	"Name": "Unknown",
	"Colors": ["Calico, White"],
	"Date of Birth": "2021-4-22T00:00:00",
	"Intake Date": "2021-9-7T16:00:00"
}
```


**Extra**
During the Proposal phase I conducted an interview with an animal shelter in Helmond. The (Dutch) transcription of this conversation can be found in the document `Animal Shelter Interview.pdf`. 
