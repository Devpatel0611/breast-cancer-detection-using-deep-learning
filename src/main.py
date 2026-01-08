from flask import Flask, render_template, session, url_for,jsonify,request, redirect
from flask_sqlalchemy import SQLAlchemy;
import bcrypt
import tensorflow as tf 
import numpy as np
import cv2
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16


app = Flask(__name__)
app.secret_key = 'breast cancer detection'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

dic = {0: 'Benign', 1: 'Malignant'}

model = load_model('BreastCancer_DL_Model.h5')
model.make_predict_function()

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (224, 224))  
    img = np.stack([img] * 3, axis=-1)  
    img = img.astype(np.float32)  
    img = preprocess_input(img) 
    img = img.reshape(1, 224, 224, 3)

    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = base_model.predict(img)  
    features = features.reshape(1, -1)  
    return features

# Prediction function
def predict_label(img_path):
    img = preprocess_image(img_path)  
    prediction_prob = model.predict(img)[0] 

    
    predicted_class = 1 if prediction_prob > 0.5 else 0  
    dic = {0: 'Benign', 1: 'Malignant'}
    return dic[predicted_class]  

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'login_email' in request.form and 'login_password' in request.form:
            login_email = request.form['login_email']
            login_password = request.form['login_password']
            user = User.query.filter_by(email=login_email).first()
            if user and user.check_password(login_password):
                session['name'] = user.name
                session['email'] = user.email
                return redirect('/home')
            else:
                return render_template('index.html', error='Invalid login credentials')
        
        elif 'register_name' in request.form and 'register_email' in request.form and 'register_password' in request.form:
            name = request.form['register_name']
            email = request.form['register_email']
            password = request.form['register_password']
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            return render_template('index.html', success='Registration successful, please log in.')
        
    

    else:
        return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/consult')
def consult():
    return render_template('consult.html')

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image'] 
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return jsonify({"prediction": p, "img_path": img_path})


if __name__ == "__main__":
    app.run(debug=True)
