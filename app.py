from flask import Flask, render_template, redirect, url_for, request, flash, session
import numpy as np
import pandas as pd

from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

from sklearn.preprocessing import StandardScaler
from src.pipeline.pred_pipeline import input_data, Pred_Pipeline

app = Flask(__name__)
app.secret_key = 'credit_metrics_secure_key'  # Secret key for session and flash messages

# Configuring sqlalchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Database Model for users
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bank_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    contact_person = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(20), nullable=False)
    license_id = db.Column(db.String(50))
    password = db.Column(db.String(200), nullable=False)

    def check_password(self, password):
        try:
            print(f"Checking password: stored={self.password[:10]}... vs provided={password}")
            result = check_password_hash(self.password, password)
            print(f"Password check result: {result}")
            return result
        except Exception as e:
            print(f"Error checking password: {e}")
            return False


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for legal page
@app.route('/legal')
def legal():
    return render_template('legal.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if user is already logged in
    if "email" in session:
        return redirect(url_for('dashboard'))
    
    # Handle login form submission
    if request.method == 'POST':
        print("Login form submitted")
        email = request.form.get('email')
        password = request.form.get('password')
        
        print(f"Login attempt with email: {email}")
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()
        if user:
            print(f"User found: {user.bank_name}")
            password_check = user.check_password(password)
            print(f"Password check result: {password_check}")
            
            if password_check:
                session['email'] = email
                session['bank_name'] = user.bank_name
                print(f"Login successful for {user.bank_name}")
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                print("Password doesn't match")
                flash('Invalid email or password', 'error')
        else:
            print(f"No user found with email: {email}")
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')


# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        bank_name = request.form.get('bank_name')
        email = request.form.get('email')
        contact_person = request.form.get('contact_person')
        contact_number = request.form.get('contact_number')
        license_id = request.form.get('license_id')
        password = request.form.get('password')

        # Basic validation
        if not bank_name or not email or not contact_person or not contact_number or not password:
            flash('Please fill in all required fields', 'error')
            return render_template('signup.html')

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please use a different email or login.', 'error')
            return render_template('signup.html')

        # Create new user with hashed password
        hashed_password = generate_password_hash(password)
        new_user = User(
            bank_name=bank_name,
            email=email,
            contact_person=contact_person,
            contact_number=contact_number,
            license_id=license_id,
            password=hashed_password
        )
        
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


# Route for dashboard page
@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if "email" not in session:
        flash('Please login to access the dashboard', 'error')
        return redirect(url_for('login'))
    
    return render_template('dashboard.html')


# Route for logout
@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('bank_name', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))


# Route for makepred page
@app.route('/makepredictions')
def makepredictions():
    return render_template('makepredictions.html')

# Route for analyze product page
@app.route('/analyzeproduct')
def analyzeproduct():
    # Check if user is logged in
    if "email" not in session:
        flash('Please login to access the product tutorials', 'error')
        return redirect(url_for('login'))
    
    return render_template('analyzeproduct.html')

# Route to see past records page
@app.route('/records')
def records():
    # Check if user is logged in
    if "email" not in session:
        flash('Please login to access your records', 'error')
        return redirect(url_for('login'))
    
    return render_template('records.html')



# Route for tutorial pages
@app.route('/tutorial/<int:page>')
def tutorial(page):
    # Check if user is logged in
    if "email" not in session:
        flash('Please login to access the tutorials', 'error')
        return redirect(url_for('login'))
    
    # Validate page number
    if page < 1 or page > 3:
        flash('Tutorial page not found', 'error')
        return redirect(url_for('analyzeproduct'))
    
    # Render the appropriate tutorial page based on the page parameter
    return render_template(f'tutorial{page}.html')

# Route for results page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('results.html')
    else:
        data = input_data(
            Age=int(request.form.get('Age', 0)),
            Income=float(request.form.get('Income', 0.0)),
            Home=request.form.get('Home', ''),
            Emp_length=float(request.form.get('Emp_length', 0.0)),
            Intent=request.form.get('Intent', ''),
            Amount=float(request.form.get('Amount', 0.0)),
            Rate=float(request.form.get('Rate', 0.0)),
            Status=request.form.get('Status', ''),
            Percent_income=float(request.form.get('Percent_income', 0.0)),
            Cred_length=int(request.form.get('Cred_length', 0))
        )
        pred_data = data.transfrom_data_as_dataframe()
        print(pred_data)
        print("Before Prediction")

        predict_pipeline = Pred_Pipeline()
        print("During Prediction")
        results, probability, message = predict_pipeline.predict(pred_data)
        print("After Prediction")
        
        # Use the message directly from the prediction pipeline
        return render_template('results.html', results=message)

# Custom 404 error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Serve component HTML files
@app.route('/components/<component_name>')
def serve_component(component_name):
    return render_template(f'components/{component_name}')

# # Route for contact page for logged-in users
# @app.route('/contact_loggedin', methods=['GET', 'POST'])
# def contact_loggedin():
#     # Check if user is logged in
#     if "email" not in session:
#         flash('Please login to access the contact page', 'error')
#         return redirect(url_for('login'))
    
#     if request.method == 'POST':
#         name = request.form.get('name')
#         email = request.form.get('email')
#         message = request.form.get('message')
        
#         # Here you can add logic to handle the contact form submission,
#         # such as sending an email or saving the message to a database.
        
#         flash('Your message has been sent successfully!', 'success')
#         return redirect(url_for('contact_loggedin'))
    
#     return render_template('contact_loggedin.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 