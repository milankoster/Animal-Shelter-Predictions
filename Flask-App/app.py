from flask import Flask, render_template, request
import joblib
import json
import numpy
from datetime import datetime

app = Flask(__name__)

model = None
label_encoder, one_hot_encoder, name_encoder = None, None, None
min_max_scaler, standard_scaler = None, None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['POST'])
def predict():
    request_json = request.get_json()

    encoded_categories = one_hot_encode(request_json)
    encoded_colors = parse_colors(request_json.get('Colors'))
    scaled_data = parse_dates(request_json.get('Date of Birth'), request_json.get('Intake Date'))

    model_data = numpy.append(encoded_categories, encoded_colors)
    model_data = numpy.append(model_data, scaled_data)
    model_data = remove_duplicates(model_data)

    prediction = model.predict([model_data])
    prediction = label_encoder.inverse_transform(prediction)

    response = json.dumps({'response': prediction[0]})

    return response, 200


def one_hot_encode(animal):
    animal_type = animal.get('Animal Type')
    breed = animal.get('Breed')
    gender = animal.get('Gender')
    intake_type = animal.get('Intake Type')
    intake_condition = animal.get('Intake Condition')
    castration_intake = animal.get('Castration Intake')
    castration_outcome = animal.get('Castration Current')

    values = one_hot_encoder.transform(
        [[animal_type, breed, gender, intake_type, intake_condition, castration_intake, castration_outcome]])
    name = name_encoder.transform([[animal.get('Name')]])

    return numpy.append(values, name)


def parse_colors(colors):
    possible_colors = ['tabby', 'tricolor', 'brown', 'black', 'white', 'orange',
                       'tortie', 'calico', 'blue', 'tan', 'brindle']
    colors = [x.lower() for x in colors]
    color_array = []
    for color in possible_colors:
        color_array.append(1) if color in colors else color_array.append(0)

    return color_array


def parse_dates(date_of_birth, intake_datetime):
    dob = datetime.strptime(date_of_birth, '%Y-%m-%dT%H:%M:%S')
    intake = datetime.strptime(intake_datetime, '%Y-%m-%dT%H:%M:%S')

    age = get_age(dob, intake)
    standard_times = [dob.year, intake.year]
    min_max_times = [dob.month, dob.isocalendar().week, dob.day, dob.weekday(),
                     intake.month, intake.isocalendar().week, intake.day, intake.weekday(), intake.hour, age]

    return scale_dates(standard_times, min_max_times)


def scale_dates(standard_times, min_max_times):
    standard_times = standard_scaler.transform([standard_times])[0]
    min_max_times = min_max_scaler.transform([min_max_times])[0]
    dob = min_max_times[:4]
    intake = min_max_times[4:]

    dob_year = standard_times[0]
    intake_year = standard_times[1]
    scaled_data = numpy.append(dob_year, dob)
    scaled_data = numpy.append(scaled_data, intake_year)
    scaled_data = numpy.append(scaled_data, intake)

    return scaled_data


def get_age(date_of_birth, intake_datetime):
    age = intake_datetime - date_of_birth
    age = age.days
    if age < 0:
        age = 0
    return age


# This should not be necessary, but particular columns were dropped during the Modelling phase
# These columns act as redundant data e.g. male or female
def remove_duplicates(model_data):
    indexes = sorted([0, 18, 35, 37], reverse=True)
    return numpy.delete(model_data, indexes)


def load_model():
    with open("models/random_forest.pkl", "rb") as file:
        loaded_model = joblib.load(file)
    return loaded_model


def load_transformers():
    with open("models/label_encoder.pkl", "rb") as file:
        label_enc = joblib.load(file)
    with open("models/name_encoder.pkl", "rb") as file:
        name_enc = joblib.load(file)
    with open("models/one_hot_encoder.pkl", "rb") as file:
        ohe = joblib.load(file)
    with open("models/min_max_scaler.pkl", "rb") as file:
        mm_scaler = joblib.load(file)
    with open("models/standard_scaler.pkl", "rb") as file:
        st_scaler = joblib.load(file)

    return label_enc, ohe, name_enc, mm_scaler, st_scaler


if __name__ == '__main__':
    model = load_model()
    label_encoder, one_hot_encoder, name_encoder, min_max_scaler, standard_scaler = load_transformers()
    app.run(debug=True)
