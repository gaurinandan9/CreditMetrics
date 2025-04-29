from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import os
import sqlite3

'''
This script helps migrate the database from the old schema to the new schema.
It creates a new database with the updated User model and transfers any existing users.
'''

# Create a temporary app instance
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_new.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# New User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bank_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    contact_person = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(20), nullable=False)
    license_id = db.Column(db.String(50))
    password = db.Column(db.String(200), nullable=False)

def migrate():
    # Check if old database exists
    if not os.path.exists('instance/users.db'):
        print("No existing database found. Creating new database.")
        with app.app_context():
            db.create_all()
        return
    
    # Create new database with updated schema
    with app.app_context():
        db.create_all()
    
    # Try to migrate data from old database to new one
    try:
        conn_old = sqlite3.connect('instance/users.db')
        cursor_old = conn_old.cursor()
        
        # Get existing users
        cursor_old.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if cursor_old.fetchone() is not None:
            cursor_old.execute("SELECT * FROM user")
            old_users = cursor_old.fetchall()
            
            # Get column names
            cursor_old.execute("PRAGMA table_info(user)")
            columns = [col[1] for col in cursor_old.fetchall()]
            
            # Print columns and data for debugging
            print(f"Old database columns: {columns}")
            print(f"Found {len(old_users)} users in old database")
            
            # Transfer users to new database
            for old_user in old_users:
                # For existing users, set placeholders for new required fields
                user_dict = dict(zip(columns, old_user))
                
                new_user = User(
                    bank_name=user_dict.get('username', 'Migrated Bank'),
                    email=user_dict.get('email', f"{user_dict.get('username', 'unknown')}@example.com"),
                    contact_person="Migrated User",
                    contact_number="0000000000",
                    license_id="",
                    password=user_dict.get('password', generate_password_hash('changeme'))
                )
                
                with app.app_context():
                    db.session.add(new_user)
                    db.session.commit()
                
            print(f"Successfully migrated {len(old_users)} users to new database")
        else:
            print("No user table found in old database")
            
        conn_old.close()
        
    except Exception as e:
        print(f"Error during migration: {e}")
        return
    
    print("Migration completed successfully.")
    print("")
    print("To complete the migration:")
    print("1. Rename the current database: mv instance/users.db instance/users.db.backup")
    print("2. Rename the new database: mv instance/users_new.db instance/users.db")
    print("3. Run the application: python app.py")

if __name__ == "__main__":
    migrate() 